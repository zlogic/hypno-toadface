mod display;
mod gpu;
mod graphics;

use graphics::Hardware;

use crate::display::Surface;

fn main() {
    // TODO: make this configurable.
    const FILE: &str = "/dev/dri/card1";
    const HARDWARE: Hardware = Hardware::CPU;

    let mut surface = Surface::open(FILE).unwrap();

    let renderer = graphics::new_renderer(surface.parameters(), HARDWARE).unwrap();

    println!(
        "Using device: DRM: {}, renderer: {}",
        surface.device_name(),
        renderer.device_name()
    );

    surface.render_loop(renderer).unwrap();
}
