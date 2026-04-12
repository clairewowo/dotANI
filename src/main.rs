use std::io::Write;
use std::path::PathBuf;

use chrono::Local;
use clap::{value_parser, Arg, ArgAction, Command};
use env_logger::{Builder, Target};
use log::LevelFilter;

use dotani::{dist, params, sketch, sketch_cuda, types};

fn init_log() {
    Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%d-%H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(None, LevelFilter::Info)
        .target(Target::Stdout)
        .init();
}

fn main() {
    init_log();

    let sketch_cmd = Command::new(params::CMD_SKETCH)
        .about("Sketch genome FASTA files into DotANI sketches")
        .arg(
            Arg::new("path")
                .short('p')
                .long("path")
                .help("Input folder path containing .fna/.fa/.fasta files")
                .required(true)
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("out")
                .short('o')
                .long("out")
                .help("Output sketch file")
                .required(true)
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("thread")
                .short('t')
                .long("thread")
                .help("Number of threads used for computation")
                .default_value("16")
                .value_parser(value_parser!(u8))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("sketch_method")
                .short('m')
                .long("sketch-method")
                .help("Sketch method")
                .default_value("t1ha2")
                .value_parser(value_parser!(String))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("canonical")
                .short('C')
                .long("canonical")
                .help("Whether to use canonical k-mers")
                .default_value("true")
                .value_parser(value_parser!(bool))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ksize")
                .short('k')
                .long("ksize")
                .help("k-mer size for sketching")
                .default_value("21")
                .value_parser(value_parser!(u8))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("seed")
                .short('S')
                .long("seed")
                .help("Hash seed")
                .default_value("123")
                .value_parser(value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("scaled")
                .short('s')
                .long("scaled")
                .help("Scaled factor for FracMinHash")
                .default_value("1")
                .value_parser(value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("hv_d")
                .short('d')
                .long("hv-d")
                .help("Dimension for hypervector")
                .default_value("4096")
                .value_parser(value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("quant_scale")
                .short('Q')
                .long("quant-scale")
                .help("Scaling factor for HV quantization")
                .default_value("1.0")
                .value_parser(value_parser!(f32))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ani_th")
                .short('a')
                .long("ani-th")
                .help("ANI threshold")
                .default_value("85.0")
                .value_parser(value_parser!(f32))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("device")
                .short('D')
                .long("device")
                .help("Device to run on")
                .default_value("cpu")
                .value_parser(["cpu", "gpu"])
                .action(ArgAction::Set),
        );

    let dist_cmd = Command::new(params::CMD_DIST)
        .about("Estimate ANI from reference and query sketch files")
        .arg(
            Arg::new("path_r")
                .short('r')
                .long("path-r")
                .help("Path to reference sketch file")
                .required(true)
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("path_q")
                .short('q')
                .long("path-q")
                .help("Path to query sketch file")
                .required(true)
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("out")
                .short('o')
                .long("out")
                .help("Output ANI results file")
                .required(true)
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("thread")
                .short('t')
                .long("thread")
                .help("Number of threads used for computation")
                .default_value("16")
                .value_parser(value_parser!(u8))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("sketch_method")
                .short('m')
                .long("sketch-method")
                .help("Sketch method")
                .default_value("fracminhash")
                .value_parser(value_parser!(String))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("canonical")
                .short('C')
                .long("canonical")
                .help("Whether to use canonical k-mers")
                .default_value("true")
                .value_parser(value_parser!(bool))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ksize")
                .short('k')
                .long("ksize")
                .help("k-mer size for sketching")
                .default_value("21")
                .value_parser(value_parser!(u8))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("seed")
                .short('S')
                .long("seed")
                .help("Hash seed")
                .default_value("123")
                .value_parser(value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("scaled")
                .short('s')
                .long("scaled")
                .help("Scaled factor for FracMinHash")
                .default_value("1500")
                .value_parser(value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("hv_d")
                .short('d')
                .long("hv-d")
                .help("Dimension for hypervector")
                .default_value("4096")
                .value_parser(value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("quant_scale")
                .short('Q')
                .long("quant-scale")
                .help("Scaling factor for HV quantization")
                .default_value("1.0")
                .value_parser(value_parser!(f32))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ani_th")
                .short('a')
                .long("ani-th")
                .help("ANI threshold")
                .default_value("85.0")
                .value_parser(value_parser!(f32))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("device")
                .short('D')
                .long("device")
                .help("Device to run on")
                .default_value("cpu")
                .value_parser(["cpu", "gpu"])
                .action(ArgAction::Set),
        );

    let search_cmd = Command::new(params::CMD_SEARCH)
        .about("Search a query sketch against a sketch database")
        .arg(
            Arg::new("path")
                .short('p')
                .long("path")
                .help("Path to sketch file or sketch database")
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("path_r")
                .short('r')
                .long("path-r")
                .help("Path to reference sketch file")
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("path_q")
                .short('q')
                .long("path-q")
                .help("Path to query sketch file")
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("out")
                .short('o')
                .long("out")
                .help("Output search results file")
                .value_parser(value_parser!(PathBuf))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("thread")
                .short('t')
                .long("thread")
                .help("Number of threads used for computation")
                .default_value("16")
                .value_parser(value_parser!(u8))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("sketch_method")
                .short('m')
                .long("sketch-method")
                .help("Sketch method")
                .default_value("fracminhash")
                .value_parser(value_parser!(String))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("canonical")
                .short('C')
                .long("canonical")
                .help("Whether to use canonical k-mers")
                .default_value("true")
                .value_parser(value_parser!(bool))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ksize")
                .short('k')
                .long("ksize")
                .help("k-mer size for sketching")
                .default_value("21")
                .value_parser(value_parser!(u8))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("seed")
                .short('S')
                .long("seed")
                .help("Hash seed")
                .default_value("123")
                .value_parser(value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("scaled")
                .short('s')
                .long("scaled")
                .help("Scaled factor for FracMinHash")
                .default_value("1500")
                .value_parser(value_parser!(u64))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("hv_d")
                .short('d')
                .long("hv-d")
                .help("Dimension for hypervector")
                .default_value("4096")
                .value_parser(value_parser!(usize))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("quant_scale")
                .short('Q')
                .long("quant-scale")
                .help("Scaling factor for HV quantization")
                .default_value("1.0")
                .value_parser(value_parser!(f32))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("ani_th")
                .short('a')
                .long("ani-th")
                .help("ANI threshold")
                .default_value("85.0")
                .value_parser(value_parser!(f32))
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("device")
                .short('D')
                .long("device")
                .help("Device to run on")
                .default_value("cpu")
                .value_parser(["cpu", "gpu"])
                .action(ArgAction::Set),
        );

    let matches = Command::new("dotani")
        .version(params::VERSION)
        .about("DotANI: Fast and memory-efficient ANI estimation in hyperdimensional space")
        .arg_required_else_help(true)
        .subcommand_required(true)
        .subcommand(sketch_cmd)
        .subcommand(dist_cmd)
        .subcommand(search_cmd)
        .get_matches();

    if let Some(sketch_m) = matches.subcommand_matches(params::CMD_SKETCH) {
        let cli_params = types::CliParams {
            mode: params::CMD_SKETCH.to_string(),
            path: sketch_m.get_one::<PathBuf>("path").cloned().unwrap(),
            path_ref_sketch: PathBuf::new(),
            path_query_sketch: PathBuf::new(),
            out_file: sketch_m.get_one::<PathBuf>("out").cloned().unwrap(),
            ksize: *sketch_m.get_one::<u8>("ksize").unwrap(),
            sketch_method: sketch_m
                .get_one::<String>("sketch_method")
                .cloned()
                .unwrap(),
            canonical: *sketch_m.get_one::<bool>("canonical").unwrap(),
            seed: *sketch_m.get_one::<u64>("seed").unwrap(),
            scaled: *sketch_m.get_one::<u64>("scaled").unwrap(),
            hv_d: *sketch_m.get_one::<usize>("hv_d").unwrap(),
            hv_quant_scale: *sketch_m.get_one::<f32>("quant_scale").unwrap(),
            ani_threshold: *sketch_m.get_one::<f32>("ani_th").unwrap(),
            if_compressed: true,
            threads: *sketch_m.get_one::<u8>("thread").unwrap(),
            device: sketch_m.get_one::<String>("device").cloned().unwrap(),
        };

        rayon::ThreadPoolBuilder::new()
            .num_threads(cli_params.threads as usize)
            .build_global()
            .unwrap();

        let sketch_params = types::SketchParams::new(&cli_params);

        if sketch_params.device == "gpu" {
            sketch_cuda::sketch_cuda(sketch_params);
        } else {
            sketch::sketch(sketch_params);
        }
    } else if let Some(dist_m) = matches.subcommand_matches(params::CMD_DIST) {
        let cli_params = types::CliParams {
            mode: params::CMD_DIST.to_string(),
            path: PathBuf::new(),
            path_ref_sketch: dist_m.get_one::<PathBuf>("path_r").cloned().unwrap(),
            path_query_sketch: dist_m.get_one::<PathBuf>("path_q").cloned().unwrap(),
            out_file: dist_m.get_one::<PathBuf>("out").cloned().unwrap(),
            ksize: *dist_m.get_one::<u8>("ksize").unwrap(),
            sketch_method: dist_m
                .get_one::<String>("sketch_method")
                .cloned()
                .unwrap(),
            canonical: *dist_m.get_one::<bool>("canonical").unwrap(),
            seed: *dist_m.get_one::<u64>("seed").unwrap(),
            scaled: *dist_m.get_one::<u64>("scaled").unwrap(),
            hv_d: *dist_m.get_one::<usize>("hv_d").unwrap(),
            hv_quant_scale: *dist_m.get_one::<f32>("quant_scale").unwrap(),
            ani_threshold: *dist_m.get_one::<f32>("ani_th").unwrap(),
            if_compressed: true,
            threads: *dist_m.get_one::<u8>("thread").unwrap(),
            device: dist_m.get_one::<String>("device").cloned().unwrap(),
        };

        rayon::ThreadPoolBuilder::new()
            .num_threads(cli_params.threads as usize)
            .build_global()
            .unwrap();

        let mut sketch_dist = types::SketchDist::new(&cli_params);
        dist::dist(&mut sketch_dist);
    } else if let Some(search_m) = matches.subcommand_matches(params::CMD_SEARCH) {
        let cli_params = types::CliParams {
            mode: params::CMD_SEARCH.to_string(),
            path: search_m
                .get_one::<PathBuf>("path")
                .cloned()
                .unwrap_or_default(),
            path_ref_sketch: search_m
                .get_one::<PathBuf>("path_r")
                .cloned()
                .unwrap_or_default(),
            path_query_sketch: search_m
                .get_one::<PathBuf>("path_q")
                .cloned()
                .unwrap_or_default(),
            out_file: search_m
                .get_one::<PathBuf>("out")
                .cloned()
                .unwrap_or_default(),
            ksize: *search_m.get_one::<u8>("ksize").unwrap(),
            sketch_method: search_m
                .get_one::<String>("sketch_method")
                .cloned()
                .unwrap(),
            canonical: *search_m.get_one::<bool>("canonical").unwrap(),
            seed: *search_m.get_one::<u64>("seed").unwrap(),
            scaled: *search_m.get_one::<u64>("scaled").unwrap(),
            hv_d: *search_m.get_one::<usize>("hv_d").unwrap(),
            hv_quant_scale: *search_m.get_one::<f32>("quant_scale").unwrap(),
            ani_threshold: *search_m.get_one::<f32>("ani_th").unwrap(),
            if_compressed: true,
            threads: *search_m.get_one::<u8>("thread").unwrap(),
            device: search_m.get_one::<String>("device").cloned().unwrap(),
        };

        rayon::ThreadPoolBuilder::new()
            .num_threads(cli_params.threads as usize)
            .build_global()
            .unwrap();

        // TODO: support search
        let _ = cli_params;
    } else {
        unreachable!("clap should ensure we don't get here");
    }
}