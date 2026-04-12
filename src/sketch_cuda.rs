use crate::types::*;

#[cfg(feature = "cuda-sketch")]
use {
    crate::{dist, fastx_reader, hd, utils},
    cudarc::{
        driver::{CudaContext, LaunchConfig, PushKernelArg},
        nvrtc::Ptx,
    },
    glob::glob,
    log::info,
    rayon::prelude::*,
    std::cmp::max,
    std::collections::HashSet,
    std::path::{Path, PathBuf},
    std::sync::Arc,
};

#[cfg(feature = "cuda-sketch")]
const CUDA_KERNEL_MY_STRUCT: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda_kmer_hash.ptx"));

#[cfg(feature = "cuda-sketch")]
const SEQ_NT4_TABLE: [u8; 256] = [
    0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
];

#[allow(unused_variables)]
#[cfg(not(feature = "cuda-sketch"))]
pub fn sketch_cuda(params: SketchParams) {
    use log::error;

    error!("Cuda sketching is not supported. Please add `--features cuda-sketch` for installation to enable it.");
}

//  Sketch function to sketch all .fna files in folder path
#[cfg(all(target_arch = "x86_64", feature = "cuda-sketch"))]
pub fn sketch_cuda(params: SketchParams) {
    let files = utils::get_fasta_files(&params.path);
    let n_file = files.len();

    info!("Start GPU sketching...");
    let pb = utils::get_progress_bar(n_file);

    let ctx = Arc::new(CudaContext::new(0).unwrap());
    let module = Arc::new(
        ctx.load_module(Ptx::from_src(CUDA_KERNEL_MY_STRUCT))
            .unwrap(),
    );

    let mut all_filesketch: Vec<FileSketch> = (0..n_file)
        .map(|i| FileSketch {
            ksize: params.ksize,
            scaled: params.scaled,
            seed: params.seed,
            canonical: params.canonical,
            hv_d: params.hv_d,
            hv_quant_bits: 16u8,
            hv_norm_2: 0,
            file_str: files[i].display().to_string(),
            hv: Vec::<i16>::new(),
        })
        .collect();

    all_filesketch.par_iter_mut().for_each(|sketch| {
        let kmer_hash_set = extract_kmer_t1ha2_cuda(sketch, &ctx, &module);

        let hv = if is_x86_feature_detected!("avx2") {
            unsafe { hd::encode_hash_hd_avx2(&kmer_hash_set, &sketch) }
        } else {
            hd::encode_hash_hd(&kmer_hash_set, &sketch)
        };

        sketch.hv_norm_2 = dist::compute_hv_l2_norm(&hv);

        if params.if_compressed {
            sketch.hv_quant_bits = unsafe { hd::compress_hd_sketch(sketch, &hv) };
        }

        pb.inc(1);
        pb.eta();
    });

    pb.finish_and_clear();

    info!(
        "Sketching {} files took {:.2}s - Speed: {:.1} files/s",
        files.len(),
        pb.elapsed().as_secs_f32(),
        pb.per_sec()
    );

    utils::dump_sketch(&all_filesketch, &params.out_file);
}

#[cfg(feature = "cuda-sketch")]
fn extract_kmer_t1ha2_cuda(
    sketch: &FileSketch,
    ctx: &Arc<CudaContext>,
    module: &Arc<cudarc::driver::CudaModule>,
) -> HashSet<u64> {
    let fna_file = PathBuf::from(sketch.file_str.clone());
    let fna_seqs = fastx_reader::read_merge_seq(&fna_file);

    let n_bps = fna_seqs.len();
    let ksize = sketch.ksize as usize;
    let canonical = sketch.canonical;
    let scaled = sketch.scaled;
    let seed = sketch.seed;

    if n_bps < ksize {
        return HashSet::new();
    }

    let n_kmers = n_bps - ksize + 1;
    let kmer_per_thread = 512usize;
    let n_threads = n_kmers.div_ceil(kmer_per_thread);

    let stream = ctx.default_stream();
    let gpu_seq = stream.clone_htod(&fna_seqs).unwrap();

    let n_hash_per_thread = max(kmer_per_thread / sketch.scaled as usize * 4, 8);
    let n_hash_array = n_hash_per_thread * n_threads;
    let mut gpu_kmer_hash = stream.alloc_zeros::<u64>(n_hash_array).unwrap();

    let f = module.load_function("cuda_kmer_t1ha2").unwrap();
    let mut builder = stream.launch_builder(&f);
    builder.arg(&gpu_seq);
    builder.arg(&n_bps);
    builder.arg(&kmer_per_thread);
    builder.arg(&n_hash_per_thread);
    builder.arg(&ksize);
    let binding = u64::MAX / scaled;
    builder.arg(&binding);
    builder.arg(&seed);
    builder.arg(&canonical);
    builder.arg(&mut gpu_kmer_hash);

    unsafe {
        builder
            .launch(LaunchConfig::for_num_elems(n_threads as u32))
            .unwrap();
    }

    let host_kmer_hash = stream.clone_dtoh(&gpu_kmer_hash).unwrap();

    host_kmer_hash.into_iter().filter(|&h| h != 0).collect()
}

#[cfg(feature = "cuda-sketch")]
pub fn cuda_mmhash_bitpack_parallel(
    path_fna: &String,
    ksize: usize,
    canonical: bool,
    scaled: u64,
) -> Vec<HashSet<u64>> {
    let files = utils::get_fasta_files(&PathBuf::from(path_fna));
    let n_file = files.len();
    let pb = utils::get_progress_bar(n_file);

    let ctx = Arc::new(CudaContext::new(0).unwrap());
    let module = Arc::new(
        ctx.load_module(Ptx::from_src(CUDA_KERNEL_MY_STRUCT))
            .unwrap(),
    );

    let index_vec: Vec<usize> = (0..files.len()).collect();
    let sketch_kmer_sets: Vec<HashSet<u64>> = index_vec
        .par_iter()
        .map(|&i| {
            let fna_seqs = fastx_reader::read_merge_seq(&files[i]);

            let n_bps = fna_seqs.len();
            if n_bps < ksize {
                pb.inc(1);
                return HashSet::new();
            }

            let n_kmers = n_bps - ksize + 1;
            let bp_per_thread = 512usize;
            let n_threads = n_kmers.div_ceil(bp_per_thread);

            let stream = ctx.default_stream();
            let gpu_seq = stream.clone_htod(&fna_seqs).unwrap();
            let gpu_seq_nt4_table = stream.clone_htod(&SEQ_NT4_TABLE).unwrap();

            let n_hash_per_thread = max(bp_per_thread / scaled as usize * 3, 8);
            let n_hash_array = n_hash_per_thread * n_threads;
            let mut gpu_kmer_bit_hash = stream.alloc_zeros::<u64>(n_hash_array).unwrap();

            let f = module.load_function("cuda_kmer_bit_pack_mmhash").unwrap();
            let mut builder = stream.launch_builder(&f);
            builder.arg(&gpu_seq);
            builder.arg(&n_bps);
            builder.arg(&bp_per_thread);
            builder.arg(&n_hash_per_thread);
            builder.arg(&ksize);
            let binding = u64::MAX / scaled;
            builder.arg(&binding);
            builder.arg(&canonical);
            builder.arg(&gpu_seq_nt4_table);
            builder.arg(&mut gpu_kmer_bit_hash);

            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n_threads as u32))
                    .unwrap();
            }

            let host_kmer_bit_hash = stream.clone_dtoh(&gpu_kmer_bit_hash).unwrap();

            pb.inc(1);
            host_kmer_bit_hash.into_iter().filter(|&h| h != 0).collect()
        })
        .collect();

    pb.finish_and_clear();
    sketch_kmer_sets
}

#[cfg(feature = "cuda-sketch")]

pub fn cuda_t1ha2_hash_parallel(
    path_fna: &String,
    ksize: usize,
    canonical: bool,
    scaled: u64,
    seed: u64,
) -> Vec<HashSet<u64>> {
    let files: Vec<_> = glob(Path::new(path_fna).join("*.fna").to_str().unwrap())
        .expect("Failed to read glob pattern")
        .collect();

    let n_file = files.len();
    let pb = utils::get_progress_bar(n_file);

    let ctx = Arc::new(CudaContext::new(0).unwrap());
    let module = Arc::new(
        ctx.load_module(Ptx::from_src(CUDA_KERNEL_MY_STRUCT))
            .unwrap(),
    );

    let index_vec: Vec<usize> = (0..files.len()).collect();
    let sketch_kmer_sets: Vec<HashSet<u64>> = index_vec
        .par_iter()
        .map(|i| {
            let fna_seqs = fastx_reader::read_merge_seq(files[*i].as_ref().unwrap());

            let n_bps = fna_seqs.len();
            if n_bps < ksize {
                pb.inc(1);
                return HashSet::new();
            }

            let n_kmers = n_bps - ksize + 1;
            let kmer_per_thread = 512usize;
            let n_threads = n_kmers.div_ceil(kmer_per_thread);

            let stream = ctx.default_stream();
            let gpu_seq = stream.clone_htod(&fna_seqs).unwrap();

            let n_hash_per_thread = max(kmer_per_thread / scaled as usize * 3, 8);
            let n_hash_array = n_hash_per_thread * n_threads;
            let mut gpu_kmer_hash = stream.alloc_zeros::<u64>(n_hash_array).unwrap();

            let f = module.load_function("cuda_kmer_t1ha2").unwrap();
            let mut builder = stream.launch_builder(&f);
            builder.arg(&gpu_seq);
            builder.arg(&n_bps);
            builder.arg(&kmer_per_thread);
            builder.arg(&n_hash_per_thread);
            builder.arg(&ksize);
            let binding = u64::MAX / scaled;
            builder.arg(&binding);
            builder.arg(&seed);
            builder.arg(&canonical);
            builder.arg(&mut gpu_kmer_hash);

            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n_threads as u32))
                    .unwrap();
            }

            let host_kmer_hash = stream.clone_dtoh(&gpu_kmer_hash).unwrap();

            pb.inc(1);
            host_kmer_hash.into_iter().filter(|&h| h != 0).collect()
        })
        .collect();

    pb.finish_and_clear();
    sketch_kmer_sets
}