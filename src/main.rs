use std::{
    env::{self},
    fs::File,
    io::{self, BufReader, Read, Write},
    process::exit,
    time::Instant,
};

use crate::graphics::Scene;

mod gpu;
mod graphics;
mod sighandler;
mod sound;

#[derive(Debug)]
pub struct Args {
    speed: f64,
    shader_filename: Option<String>,
    no_print_fps: bool,
    keep_last_frame: bool,
    sound_device: Option<String>,
}

const USAGE_INSTRUCTIONS: &str = "Usage: hypno-toadface [OPTIONS] [--sound=<DEVICEPATH>]\n\n\
Options:\
\n      --speed=<SPEED>          Animation speed [default: 0.04]\
\n      --shader-file=<FILENAME> Filename for custom SPIR-V frag shader [default: uses built-in]\
\n      --no-print-fps           If specified, don't print FPS counter\
\n      --keep-last-frame        If specified, save last frame and provide it to the shader\
\n      --help                   Print help";

fn main() {
    let mut args = Args {
        speed: 0.04,
        shader_filename: None,
        no_print_fps: false,
        keep_last_frame: false,
        sound_device: None,
    };
    for arg in env::args().skip(1) {
        if arg.starts_with("--") {
            // Option flags.
            if arg == "--help" {
                println!("{}", USAGE_INSTRUCTIONS);
                exit(0);
            }
            if arg == "--no-print-fps" {
                args.no_print_fps = true;
                continue;
            }
            if arg == "--keep-last-frame" {
                args.keep_last_frame = true;
                continue;
            }
            let (name, value) = if let Some(arg) = arg.split_once('=') {
                arg
            } else {
                eprintln!("Option flag {} has no value", arg);
                println!("{}", USAGE_INSTRUCTIONS);
                exit(2);
            };
            if name == "--speed" {
                match value.parse() {
                    Ok(speed) => args.speed = speed,
                    Err(err) => {
                        eprintln!("Speed {} has an unsupported value {}: {}", name, value, err);
                        println!("{}", USAGE_INSTRUCTIONS);
                        exit(2)
                    }
                };
            } else if name == "--shader-file" {
                args.shader_filename = Some(value.to_string());
            } else if name == "--sound" {
                args.sound_device = Some(value.to_string());
            } else {
                eprintln!("Unsupported argument {}", arg);
            }
        }
    }

    run_animation(args)
}

fn load_file(path: &str) -> Result<Vec<u8>, io::Error> {
    let f = File::open(path)?;
    let r = BufReader::new(f);
    let shader_bytes = r.bytes().collect::<Result<Vec<_>, _>>()?;
    Ok(shader_bytes)
}

fn run_animation(args: Args) {
    let renderer = {
        let shader_data = if let Some(shader_filename) = args.shader_filename {
            match load_file(&shader_filename) {
                Ok(shader_data) => Some(shader_data),
                Err(err) => {
                    eprintln!("Failed to load shader file {}: {}", shader_filename, err);
                    println!("{}", USAGE_INSTRUCTIONS);
                    exit(2)
                }
            }
        } else {
            None
        };

        let conf = gpu::Configuration {
            shader: shader_data.as_deref(),
            store_image: args.keep_last_frame,
        };
        gpu::Gpu::init(conf).expect("Failed to init GPU")
    };
    let player = args
        .sound_device
        .map(|path| sound::Player::new(&path).expect("Failed to init audio device"));

    println!("Using video device: {}", renderer.device_name());

    let start = Instant::now();
    let mut framecounter_start = Instant::now();
    let mut framecounter_frames = 0usize;
    sighandler::listen_to_sigint();
    loop {
        if sighandler::stop_requested() {
            break;
        }
        let scene = Scene {
            timecode: start.elapsed().as_secs_f64() * args.speed,
        };

        let result = match renderer.render(&scene) {
            Ok(res) => res,
            Err(err) => {
                eprintln!("Failed to render scene: {}", err);
                continue;
            }
        };
        if result.swapchain_suboptimal {
            eprintln!("Swapchain is suboptimal");
        }
        if result.queue_suboptimal {
            eprintln!("Queue is suboptimal");
        }

        framecounter_frames += 1;
        let print_fps = !args.no_print_fps;
        if framecounter_frames > 100 && print_fps {
            let elapsed = framecounter_start.elapsed();
            let fps = framecounter_frames as f32 / elapsed.as_secs_f32();

            print!("Average FPS: {:.2}   \r", fps);
            let _ = io::stdout().flush();

            framecounter_frames = 0;
            framecounter_start = Instant::now();
        }
    }
    drop(renderer);
    drop(player);

    println!("\nGoodbye.");
}
