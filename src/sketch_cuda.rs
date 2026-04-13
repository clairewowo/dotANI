use crate::types::*;

#[cfg(feature = "cuda-sketch")]
use {
    crate::{dist, fastx_reader, hd, ultraloglog::UltraLogLog;::UltraLogLog, utils},
    cudarc::{
        driver::{CudaContext, LaunchConfig, PushKernelArg},
        nvrtc::Ptx,
    },
    glob::glob,
    log::info,
    rayon::prelude::*,
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

// Sketch all FASTA files using GPU hashing, then CPU ULL + CPU HD encoding
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

    let results: Vec<(FileSketch, Option<FileUllSketch>)> = files
        .par_iter()
        .map(|file| {
            let mut sketch = FileSketch {
                ksize: params.ksize,
                scaled: params.scaled,
                seed: params.seed,
                canonical: params.canonical,
                hv_d: params.hv_d,
                hv_quant_bits: 16u8,
                hv_norm_2: 0,
                file_str: file.display().to_string(),
                hv: Vec::<i16>::new(),
            };

            // Full hash stream from GPU
            let full_hashes = extract_kmer_t1ha2_cuda_full_hashes(&sketch, &ctx, &module);

            let threshold = u64::MAX / sketch.scaled;
            let mut sampled_hash_set = HashSet::<u64>::new();

            let ull_record = if params.if_ull {
                let mut ull =
                    UltraLogLog::new(params.ull_p).expect("Invalid UltraLogLog precision");

                for &h in &full_hashes {
                    ull.add(h);
                    if h < threshold {
                        sampled_hash_set.insert(h);
                    }
                }

                Some(FileUllSketch {
                    ksize: params.ksize,
                    canonical: params.canonical,
                    seed: params.seed,
                    ull_p: params.ull_p,
                    file_str: sketch.file_str.clone(),
                    ull_state: ull.get_state().to_vec(),
                })
            } else {
                for &h in &full_hashes {
                    if h < threshold {
                        sampled_hash_set.insert(h);
                    }
                }
                None
            };

            let hv = if is_x86_feature_detected!("avx2") {
                unsafe { hd::encode_hash_hd_avx2(&sampled_hash_set, &sketch) }
            } else {
                hd::encode_hash_hd(&sampled_hash_set, &sketch)
            };

            sketch.hv_norm_2 = dist::compute_hv_l2_norm(&hv);

            if params.if_compressed {
                sketch.hv_quant_bits = unsafe { hd::compress_hd_sketch(&mut sketch, &hv) };
            } else {
                sketch.hv = hv.clone();
            }

            pb.inc(1);
            pb.eta();

            (sketch, ull_record)
        })
        .collect();

    pb.finish_and_clear();

    let all_filesketch: Vec<FileSketch> = results.iter().map(|(fs, _)| fs.clone()).collect();
    let all_ullsketch: Vec<FileUllSketch> = results.into_iter().filter_map(|(_, u)| u).collect();

    info!(
        "Sketching {} files took {:.2}s - Speed: {:.1} files/s",
        files.len(),
        pb.elapsed().as_secs_f32(),
        pb.per_sec()
    );

    utils::dump_sketch(&all_filesketch, &params.out_file);

    if params.if_ull {
        utils::dump_ull_sketch(&all_ullsketch, &params.ull_out_file);
    }
}

#[cfg(feature = "cuda-sketch")]
fn extract_kmer_t1ha2_cuda_full_hashes(
    sketch: &FileSketch,
    ctx: &Arc<CudaContext>,
    module: &Arc<cudarc::driver::CudaModule>,
) -> Vec<u64> {
    let fna_file = PathBuf::from(sketch.file_str.clone());
    let fna_seqs = fastx_reader::read_merge_seq(&fna_file);

    let n_bps = fna_seqs.len();
    let ksize = sketch.ksize as usize;
    let canonical = sketch.canonical;
    let seed = sketch.seed;

    if n_bps < ksize {
        return Vec::new();
    }

    let n_kmers = n_bps - ksize + 1;
    let kmer_per_thread = 512usize;
    let n_threads = n_kmers.div_ceil(kmer_per_thread);

    let stream = ctx.default_stream();
    let gpu_seq = stream.clone_htod(&fna_seqs).unwrap();

    // Full hashes: enough slots for every k-mer a thread may emit
    let n_hash_per_thread = kmer_per_thread;
    let n_hash_array = n_hash_per_thread * n_threads;
    let mut gpu_kmer_hash = stream.alloc_zeros::<u64>(n_hash_array).unwrap();

    let f = module.load_function("cuda_kmer_t1ha2").unwrap();
    let mut builder = stream.launch_builder(&f);
    builder.arg(&gpu_seq);
    builder.arg(&n_bps);
    builder.arg(&kmer_per_thread);
    builder.arg(&n_hash_per_thread);
    builder.arg(&ksize);

    // Hard-coded full hash mode
    let full_threshold = u64::MAX;
    builder.arg(&full_threshold);

    builder.arg(&seed);
    builder.arg(&canonical);
    builder.arg(&mut gpu_kmer_hash);

    unsafe {
        builder
            .launch(LaunchConfig::for_num_elems(n_threads as u32))
            .unwrap();
    }

    let host_kmer_hash = stream.clone_dtoh(&gpu_kmer_hash).unwrap();

    // Zero is the empty sentinel in the output array.
    // A real hash of 0 is astronomically rare, so dropping it is acceptable here.
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

            // Full hashes from GPU, then sample on CPU
            let n_hash_per_thread = bp_per_thread;
            let n_hash_array = n_hash_per_thread * n_threads;
            let mut gpu_kmer_bit_hash = stream.alloc_zeros::<u64>(n_hash_array).unwrap();

            let f = module.load_function("cuda_kmer_bit_pack_mmhash").unwrap();
            let mut builder = stream.launch_builder(&f);
            builder.arg(&gpu_seq);
            builder.arg(&n_bps);
            builder.arg(&bp_per_thread);
            builder.arg(&n_hash_per_thread);
            builder.arg(&ksize);

            let full_threshold = u64::MAX;
            builder.arg(&full_threshold);

            builder.arg(&canonical);
            builder.arg(&gpu_seq_nt4_table);
            builder.arg(&mut gpu_kmer_bit_hash);

            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n_threads as u32))
                    .unwrap();
            }

            let host_kmer_bit_hash = stream.clone_dtoh(&gpu_kmer_bit_hash).unwrap();
            let threshold = u64::MAX / scaled;

            pb.inc(1);
            host_kmer_bit_hash
                .into_iter()
                .filter(|&h| h != 0 && h < threshold)
                .collect()
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

            // Full hashes from GPU, then sample on CPU
            let n_hash_per_thread = kmer_per_thread;
            let n_hash_array = n_hash_per_thread * n_threads;
            let mut gpu_kmer_hash = stream.alloc_zeros::<u64>(n_hash_array).unwrap();

            let f = module.load_function("cuda_kmer_t1ha2").unwrap();
            let mut builder = stream.launch_builder(&f);
            builder.arg(&gpu_seq);
            builder.arg(&n_bps);
            builder.arg(&kmer_per_thread);
            builder.arg(&n_hash_per_thread);
            builder.arg(&ksize);

            let full_threshold = u64::MAX;
            builder.arg(&full_threshold);

            builder.arg(&seed);
            builder.arg(&canonical);
            builder.arg(&mut gpu_kmer_hash);

            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n_threads as u32))
                    .unwrap();
            }

            let host_kmer_hash = stream.clone_dtoh(&gpu_kmer_hash).unwrap();
            let threshold = u64::MAX / scaled;

            pb.inc(1);
            host_kmer_hash
                .into_iter()
                .filter(|&h| h != 0 && h < threshold)
                .collect()
        })
        .collect();

    pb.finish_and_clear();
    sketch_kmer_sets
}