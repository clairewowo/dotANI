use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};

use log::{info, warn};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::{hd, types::*};

pub fn get_fasta_files(path: &PathBuf) -> Vec<PathBuf> {
    let mut all_files = Vec::new();
    for t in ["*.fna", "*.fa", "*.fasta"] {
        let mut files: Vec<_> = glob(path.join(t).to_str().unwrap())
            .expect("Failed to read glob pattern")
            .map(|f| f.unwrap())
            .collect();

        all_files.append(&mut files);
    }

    all_files
}

pub fn get_progress_bar(n_file: usize) -> ProgressBar {
    let pb = ProgressBar::new(n_file as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} ({percent}%) - Elapsed: {elapsed_precise}, ETA: {eta_precise}")
            .unwrap(),
    );

    pb
}

pub fn dump_sketch(file_sketch: &Vec<FileSketch>, out_file_path: &PathBuf) {
    let out_filename = out_file_path.to_str().unwrap();

    let serialized = bincode::serialize::<Vec<FileSketch>>(file_sketch).unwrap();
    fs::write(out_filename, &serialized).expect("Dump sketch file failed!");

    let sketch_size_mb = serialized.len() as f32 / 1024.0 / 1024.0;
    info!(
        "Dump sketch file to {} with size {:.2} MB",
        out_filename, sketch_size_mb
    );
}

pub fn load_sketch(path: &Path) -> Vec<FileSketch> {
    info!("Loading sketch from {}", path.to_str().unwrap());
    let serialized = fs::read(path).expect("Opening sketch file failed!");
    bincode::deserialize::<Vec<FileSketch>>(&serialized[..]).unwrap()
}

pub fn dump_ull_sketch(file_ull_sketch: &Vec<FileUllSketch>, out_file_path: &PathBuf) {
    let out_filename = out_file_path.to_str().unwrap();

    let serialized = bincode::serialize::<Vec<FileUllSketch>>(file_ull_sketch).unwrap();
    fs::write(out_filename, &serialized).expect("Dump ULL sketch file failed!");

    let sketch_size_mb = serialized.len() as f32 / 1024.0 / 1024.0;
    info!(
        "Dump ULL sketch file to {} with size {:.2} MB",
        out_filename, sketch_size_mb
    );
}

pub fn load_ull_sketch(path: &Path) -> Vec<FileUllSketch> {
    info!("Loading ULL sketch from {}", path.to_str().unwrap());
    let serialized = fs::read(path).expect("Opening ULL sketch file failed!");
    bincode::deserialize::<Vec<FileUllSketch>>(&serialized[..]).unwrap()
}

pub fn dump_ani_file(sketch_dist: &SketchDist) {
    let mut indices = (0..sketch_dist.file_ani.len()).collect::<Vec<_>>();
    indices.sort_by(|&i1, &i2| {
        sketch_dist.file_ani[i1]
            .1
            .partial_cmp(&sketch_dist.file_ani[i2].1)
            .unwrap()
    });
    indices.reverse();

    let mut csv_str = String::new();
    let mut cnt: f32 = 0.0;
    for i in 0..sketch_dist.file_ani.len() {
        if sketch_dist.file_ani[indices[i]].1 >= sketch_dist.ani_threshold {
            csv_str.push_str(&format!(
                "{}\t{}\t{:.3}\n",
                sketch_dist.file_ani[indices[i]].0 .0,
                sketch_dist.file_ani[indices[i]].0 .1,
                sketch_dist.file_ani[indices[i]].1
            ));
            cnt += 1.0;
        } else {
            break;
        }
    }

    fs::write(sketch_dist.out_file.to_str().unwrap(), csv_str.as_bytes())
        .expect("Dump ANI file failed!");

    let total_dist = sketch_dist.file_ani.len() as f32;
    let perc = cnt / total_dist * 100.0;
    if perc < 5.0 {
        warn!(
            "Output ANIs with threshold {:.1} are too divergent: {} of {} ({:.2}%) ANIs are reported",
            sketch_dist.ani_threshold, cnt, total_dist, perc
        );
    } else {
        info!(
            "Output {} of {} ANIs above threshold {:.1} to file {}",
            cnt,
            total_dist,
            sketch_dist.ani_threshold,
            sketch_dist.out_file.to_str().unwrap()
        )
    }
}

pub fn dump_distribution_to_txt(path: &Path) {
    let mut file_sketch = load_sketch(path);

    hd::decompress_file_sketch(&mut file_sketch);

    let data: Vec<Vec<i16>> = (0..file_sketch.len())
        .map(|i| file_sketch[i].hv.clone())
        .collect();

    let mut hist: HashMap<i16, u32> = HashMap::new();
    for row in &data {
        for v in row {
            if hist.get(v).is_none() {
                hist.insert(*v, 1);
            } else if let Some(c) = hist.get_mut(v) {
                *c += 1;
            }
        }
    }

    for kv in hist {
        println!("{}\t{}", kv.0, kv.1);
    }
}