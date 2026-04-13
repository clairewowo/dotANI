use ultraloglog::UltraLogLog;

use crate::hd;
use crate::types::*;
use crate::utils;

use log::info;
use rayon::prelude::*;

use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn dist(sketch_dist: &mut SketchDist) {
    let tstart = Instant::now();
    let if_sym = sketch_dist.path_ref_sketch == sketch_dist.path_query_sketch;

    let ref_ull_sketch = utils::load_ull_sketch(sketch_dist.path_ref_ull.as_path());
    let query_ull_sketch = if if_sym {
        ref_ull_sketch.clone()
    } else {
        utils::load_ull_sketch(sketch_dist.path_query_ull.as_path())
    };

    let mut ref_file_sketch = utils::load_sketch(sketch_dist.path_ref_sketch.as_path());
    let mut query_file_sketch = if if_sym {
        ref_file_sketch.clone()
    } else {
        utils::load_sketch(sketch_dist.path_query_sketch.as_path())
    };

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

    hd::decompress_file_sketch(&mut ref_file_sketch);
    hd::decompress_file_sketch(&mut query_file_sketch);

    compute_hv_ani(
        sketch_dist,
        &ref_file_sketch,
        &query_file_sketch,
        &ref_ull_sketch,
        &query_ull_sketch,
        ksize_ref,
        if_sym,
    );

    utils::dump_ani_file(sketch_dist);

    info!(
        "Computed ANIs for {} ref files and {} query files took {:.3}s",
        ref_file_sketch.len(),
        query_file_sketch.len(),
        tstart.elapsed().as_secs_f32()
    );
}

pub fn compute_hv_l2_norm(hv: &Vec<i32>) -> i64 {
    hv.iter()
        .map(|&num| {
            let x = num as i64;
            x * x
        })
        .sum()
}

#[inline]
pub fn ull_cardinality_from_state(state: &[u8]) -> f64 {
    let ull = UltraLogLog::wrap(state.to_vec()).expect("Invalid UltraLogLog state");
    ull.get_distinct_count_estimate()
}

#[inline]
pub fn compute_pairwise_dot(r: &[i32], q: &[i32]) -> i64 {
    r.iter()
        .zip(q.iter())
        .map(|(x, y)| (*x as i64) * (*y as i64))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_pairwise_dot_avx2(r: &[i32], q: &[i32]) -> i64 {
    assert_eq!(r.len(), q.len());

    let len = r.len();
    let n8 = len / 8;
    let mut acc: i64 = 0;

    for i in 0..n8 {
        let base = i * 8;

        let vr = unsafe { _mm256_loadu_si256(r.as_ptr().add(base) as *const __m256i) };
        let vq = unsafe { _mm256_loadu_si256(q.as_ptr().add(base) as *const __m256i) };
        let vmul = _mm256_mullo_epi32(vr, vq);

        let mut tmp = [0i32; 8];
        unsafe { _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, vmul) };

        acc += tmp.iter().map(|&x| x as i64).sum::<i64>();
    }

    for i in (n8 * 8)..len {
        acc += (r[i] as i64) * (q[i] as i64);
    }

    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_pairwise_dot_avx512(r: &[i32], q: &[i32]) -> i64 {
    assert_eq!(r.len(), q.len());

    let len = r.len();
    let n16 = len / 16;
    let mut acc: i64 = 0;

    for i in 0..n16 {
        let base = i * 16;

        let vr = unsafe { _mm512_loadu_si512(r.as_ptr().add(base) as *const __m512i) };
        let vq = unsafe { _mm512_loadu_si512(q.as_ptr().add(base) as *const __m512i) };
        let vmul = _mm512_mullo_epi32(vr, vq);

        let mut tmp = [0i32; 16];
        unsafe { _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, vmul) };

        acc += tmp.iter().map(|&x| x as i64).sum::<i64>();
    }

    for i in (n16 * 16)..len {
        acc += (r[i] as i64) * (q[i] as i64);
    }

    acc
}

pub fn compute_pairwise_ani_with_ull(
    r: &[i32],
    q: &[i32],
    card_r: f64,
    card_q: f64,
    hv_d: usize,
    ksize: u8,
) -> f32 {
    let dot = compute_pairwise_dot(r, q) as f64;
    let inter_hat = dot / hv_d as f64;

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_pairwise_ani_with_ull_avx2(
    r: &[i32],
    q: &[i32],
    card_r: f64,
    card_q: f64,
    hv_d: usize,
    ksize: u8,
) -> f32 {
    let dot = unsafe { compute_pairwise_dot_avx2(r, q) } as f64;
    let inter_hat = dot / hv_d as f64;

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_pairwise_ani_with_ull_avx512(
    r: &[i32],
    q: &[i32],
    card_r: f64,
    card_q: f64,
    hv_d: usize,
    ksize: u8,
) -> f32 {
    let dot = unsafe { compute_pairwise_dot_avx512(r, q) } as f64;
    let inter_hat = dot / hv_d as f64;

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

    let mut cnt = 0usize;
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
        .enumerate()
        .zip(index_dist.into_par_iter())
        .for_each(|((pair_idx, file_ani_pair), ind)| {
            let r = &ref_filesketch[ind.0];
            let q = &query_filesketch[ind.1];

            let card_r = ref_cards[ind.0];
            let card_q = query_cards[ind.1];

            let ani = {
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx512f") {
                        unsafe {
                            compute_pairwise_ani_with_ull_avx512(
                                &r.hv, &q.hv, card_r, card_q, r.hv_d, ksize,
                            )
                        }
                    } else if is_x86_feature_detected!("avx2") {
                        unsafe {
                            compute_pairwise_ani_with_ull_avx2(
                                &r.hv, &q.hv, card_r, card_q, r.hv_d, ksize,
                            )
                        }
                    } else {
                        compute_pairwise_ani_with_ull(
                            &r.hv, &q.hv, card_r, card_q, r.hv_d, ksize,
                        )
                    }
                }

                #[cfg(not(target_arch = "x86_64"))]
                {
                    compute_pairwise_ani_with_ull(&r.hv, &q.hv, card_r, card_q, r.hv_d, ksize)
                }
            };

            if pair_idx < 8 {
                let dot = {
                    #[cfg(target_arch = "x86_64")]
                    {
                        if is_x86_feature_detected!("avx512f") {
                            unsafe { compute_pairwise_dot_avx512(&r.hv, &q.hv) as f64 }
                        } else if is_x86_feature_detected!("avx2") {
                            unsafe { compute_pairwise_dot_avx2(&r.hv, &q.hv) as f64 }
                        } else {
                            compute_pairwise_dot(&r.hv, &q.hv) as f64
                        }
                    }

                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        compute_pairwise_dot(&r.hv, &q.hv) as f64
                    }
                };

                let inter_hat = dot / r.hv_d as f64;
                let union_hat = card_r + card_q - inter_hat;
                let jaccard = if union_hat > 0.0 {
                    inter_hat / union_hat
                } else {
                    -1.0
                };

                info!(
                    "DEBUG pair {}: {} vs {} | card_r={:.3} card_q={:.3} dot={:.3} inter_hat={:.3} union_hat={:.3} jaccard={:.6} ani={:.3}",
                    pair_idx,
                    r.file_str,
                    q.file_str,
                    card_r,
                    card_q,
                    dot,
                    inter_hat,
                    union_hat,
                    jaccard,
                    ani
                );
            }

            *file_ani_pair = ((r.file_str.clone(), q.file_str.clone()), ani);

            pb.inc(1);
            pb.eta();
        });

    pb.finish_and_clear();
}