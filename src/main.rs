use std::{
    env::{self},
    io::{self, Write},
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
    no_print_fps: bool,
    sound_device: Option<String>,
}

const USAGE_INSTRUCTIONS: &str = "Usage: hypno-toadface [OPTIONS] [--sound=<DEVICEPATH>]\n\n\
Options:\
\n      --speed=<SPEED>                  Animation speed [default: 0.04]\
\n      --no-print-fps                   If specified, don't print FPS counter\
\n      --help                           Print help";

fn main() {
    let mut args = Args {
        speed: 0.04,
        no_print_fps: false,
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
            } else if name == "--sound" {
                args.sound_device = Some(value.to_string());
            } else {
                eprintln!("Unsupported argument {}", arg);
            }
        }
    }

    run_animation(args)
}

fn run_animation(args: Args) {
    let renderer = gpu::Gpu::init().expect("Failed to init GPU");
    let player = if let Some(path) = args.sound_device {
        Some(sound::Player::new(&path).expect("Failed to init audio device"))
    } else {
        None
    };

    println!("Using video device: {}", renderer.device_name());

    let start = Instant::now();
    let mut framecounter_start = Instant::now();
    let mut framecounter_frames = 0usize;
    unsafe {
        sighandler::listen_to_sigint();
    }
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
