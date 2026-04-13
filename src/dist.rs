use ultraloglog::UltraLogLog;
use crate::hd;
use crate::types::*;
use crate::utils;
use std::time::Instant;

use log::info;
use rayon::prelude::*;

use std::arch::x86_64::*;

pub fn dist(sketch_dist: &mut SketchDist) {
    let tstart = Instant::now();
    let if_sym = sketch_dist.path_ref_sketch == sketch_dist.path_query_sketch;

    // Load ULL sketches
    let ref_ull_sketch = utils::load_ull_sketch(sketch_dist.path_ref_ull.as_path());
    let query_ull_sketch = if if_sym {
        ref_ull_sketch.clone()
    } else {
        utils::load_ull_sketch(sketch_dist.path_query_ull.as_path())
    };

    // Load HD sketches
    let mut ref_file_sketch = utils::load_sketch(sketch_dist.path_ref_sketch.as_path());
    let mut query_file_sketch = if if_sym {
        ref_file_sketch.clone()
    } else {
        utils::load_sketch(sketch_dist.path_query_sketch.as_path())
    };

    // Sanity checks
    assert_eq!(
        ref_file_sketch.len(),
        ref_ull_sketch.len(),
        "Ref HD and ULL sketch counts differ"
    );
    assert_eq!(
        query_file_sketch.len(),
        query_ull_sketch.len(),
        "Query HD and ULL sketch counts differ"
    );

    for i in 0..ref_file_sketch.len() {
        assert_eq!(
            ref_file_sketch[i].file_str,
            ref_ull_sketch[i].file_str,
            "Ref HD/ULL file order mismatch"
        );
    }
    for i in 0..query_file_sketch.len() {
        assert_eq!(
            query_file_sketch[i].file_str,
            query_ull_sketch[i].file_str,
            "Query HD/ULL file order mismatch"
        );
    }

    let ksize_ref = ref_file_sketch[0].ksize;
    let ksize_query = query_file_sketch[0].ksize;
    assert_eq!(
        ksize_ref, ksize_query,
        "Ref and query sketches use different kmer sizes!"
    );

    let hv_d_ref = ref_file_sketch[0].hv_d;
    let hv_d_query = query_file_sketch[0].hv_d;
    assert_eq!(
        hv_d_ref, hv_d_query,
        "Ref and query sketches use different HV dimensions!"
    );

    // Decompress HD sketches
    hd::decompress_file_sketch(&mut ref_file_sketch);
    hd::decompress_file_sketch(&mut query_file_sketch);

    // Compute ANI
    compute_hv_ani(
        sketch_dist,
        &ref_file_sketch,
        &query_file_sketch,
        &ref_ull_sketch,
        &query_ull_sketch,
        ksize_ref,
        if_sym,
    );

    // Dump results
    utils::dump_ani_file(sketch_dist);

    info!(
        "Computed ANIs for {} ref files and {} query files took {:.3}s",
        ref_file_sketch.len(),
        query_file_sketch.len(),
        tstart.elapsed().as_secs_f32()
    );
}

pub fn compute_hv_l2_norm(hv: &Vec<i16>) -> i32 {
    hv.iter()
        .fold(0, |sum: i32, &num| sum + (num as i32 * num as i32))
}

#[inline]
pub fn ull_cardinality_from_state(state: &[u8]) -> f64 {
    let ull = UltraLogLog::wrap(state.to_vec()).expect("Invalid UltraLogLog state");
    ull.get_distinct_count_estimate()
}

#[inline]
pub fn compute_pairwise_dot(r: &[i16], q: &[i16]) -> i32 {
    r.iter()
        .zip(q.iter())
        .map(|(x, y)| (*x as i32) * (*y as i32))
        .sum()
}

pub fn compute_pairwise_ani_with_ull(
    r: &[i16],
    q: &[i16],
    card_r: f64,
    card_q: f64,
    ksize: u8,
) -> f32 {
    let inter_hat = compute_pairwise_dot(r, q) as f64;

    if inter_hat <= 0.0 {
        return 0.0;
    }

    let union_hat = card_r + card_q - inter_hat;
    if union_hat <= 0.0 {
        return 0.0;
    }

    let jaccard = inter_hat / union_hat;
    if !jaccard.is_finite() || jaccard <= 0.0 || jaccard > 1.0 {
        return 0.0;
    }

    let ani = 1.0 + (2.0 / (1.0 / jaccard as f32 + 1.0)).ln() / (ksize as f32);

    if ani.is_nan() {
        0.0
    } else {
        ani.clamp(0.0, 1.0) * 100.0
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn compute_pairwise_dot_avx2(r: &[i16], q: &[i16]) -> i32 {
    assert_eq!(r.len(), q.len());

    let len = r.len();
    let n16 = len / 16;
    let mut dot_r_q: i32 = 0;

    for i in 0..n16 {
        let base = i * 16;

        let mm256_r = _mm256_set_epi16(
            r[base + 0],
            r[base + 1],
            r[base + 2],
            r[base + 3],
            r[base + 4],
            r[base + 5],
            r[base + 6],
            r[base + 7],
            r[base + 8],
            r[base + 9],
            r[base + 10],
            r[base + 11],
            r[base + 12],
            r[base + 13],
            r[base + 14],
            r[base + 15],
        );

        let mm256_q = _mm256_set_epi16(
            q[base + 0],
            q[base + 1],
            q[base + 2],
            q[base + 3],
            q[base + 4],
            q[base + 5],
            q[base + 6],
            q[base + 7],
            q[base + 8],
            q[base + 9],
            q[base + 10],
            q[base + 11],
            q[base + 12],
            q[base + 13],
            q[base + 14],
            q[base + 15],
        );

        let mm256_madd_32x8 = _mm256_madd_epi16(mm256_r, mm256_q);
        let mm256_madd_32x4 = _mm256_hadd_epi32(mm256_madd_32x8, _mm256_setzero_si256());

        let dot = _mm256_extract_epi32::<0>(mm256_madd_32x4)
            + _mm256_extract_epi32::<1>(mm256_madd_32x4)
            + _mm256_extract_epi32::<4>(mm256_madd_32x4)
            + _mm256_extract_epi32::<5>(mm256_madd_32x4);

        dot_r_q += dot;
    }

    // tail
    for i in (n16 * 16)..len {
        dot_r_q += r[i] as i32 * q[i] as i32;
    }

    dot_r_q
}

#[target_feature(enable = "avx2")]
pub unsafe fn compute_pairwise_ani_with_ull_avx2(
    r: &[i16],
    q: &[i16],
    card_r: f64,
    card_q: f64,
    ksize: u8,
) -> f32 {
    let inter_hat = compute_pairwise_dot_avx2(r, q) as f64;

    if inter_hat <= 0.0 {
        return 0.0;
    }

    let union_hat = card_r + card_q - inter_hat;
    if union_hat <= 0.0 {
        return 0.0;
    }

    let jaccard = inter_hat / union_hat;
    if !jaccard.is_finite() || jaccard <= 0.0 || jaccard > 1.0 {
        return 0.0;
    }

    let ani = 1.0 + (2.0 / (1.0 / jaccard as f32 + 1.0)).ln() / (ksize as f32);

    if ani.is_nan() {
        0.0
    } else {
        ani.clamp(0.0, 1.0) * 100.0
    }
}

pub fn compute_hv_ani(
    sketch_dist: &mut SketchDist,
    ref_filesketch: &Vec<FileSketch>,
    query_filesketch: &Vec<FileSketch>,
    ref_ull_sketch: &Vec<FileUllSketch>,
    query_ull_sketch: &Vec<FileUllSketch>,
    ksize: u8,
    if_symmetric: bool,
) {
    info!("Computing ANI..");

    let num_ref_files = ref_filesketch.len();
    let num_query_files = query_filesketch.len();

    let num_dists = if if_symmetric {
        num_ref_files * (num_query_files - 1) / 2
    } else {
        num_ref_files * num_query_files
    };

    let pb = utils::get_progress_bar(num_dists);

    // Compute ULL cardinalities once per genome in dist
    let ref_cards: Vec<f64> = ref_ull_sketch
        .par_iter()
        .map(|s| ull_cardinality_from_state(&s.ull_state))
        .collect();

    let query_cards: Vec<f64> = if if_symmetric {
        ref_cards.clone()
    } else {
        query_ull_sketch
            .par_iter()
            .map(|s| ull_cardinality_from_state(&s.ull_state))
            .collect()
    };

    let mut cnt = 0;
    let mut index_dist = vec![(0usize, 0usize); num_dists];
    for i in 0..num_ref_files {
        for j in (if if_symmetric { i + 1 } else { 0 })..num_query_files {
            index_dist[cnt] = (i, j);
            cnt += 1;
        }
    }

    sketch_dist.file_ani = vec![(("".to_string(), "".to_string()), 0.0); num_dists];

    sketch_dist
        .file_ani
        .par_iter_mut()
        .zip(index_dist.into_par_iter())
        .for_each(|(file_ani_pair, ind)| {
            let ani = if is_x86_feature_detected!("avx2") {
                unsafe {
                    compute_pairwise_ani_with_ull_avx2(
                        &ref_filesketch[ind.0].hv,
                        &query_filesketch[ind.1].hv,
                        ref_cards[ind.0],
                        query_cards[ind.1],
                        ksize,
                    )
                }
            } else {
                compute_pairwise_ani_with_ull(
                    &ref_filesketch[ind.0].hv,
                    &query_filesketch[ind.1].hv,
                    ref_cards[ind.0],
                    query_cards[ind.1],
                    ksize,
                )
            };

            *file_ani_pair = (
                (
                    ref_filesketch[ind.0].file_str.clone(),
                    query_filesketch[ind.1].file_str.clone(),
                ),
                ani,
            );

            pb.inc(1);
            pb.eta();
        });

    pb.finish_and_clear();
}