# Hypno Toadface

This is a weekend project, attempting to display some graphics using Linux ~DRM (Direct Rendering Manager)~ Vulkan's `VK_KHR_surface` and `VK_KHR_display` extensions.

In addition, it can also play audio using the ALSA API, without an audio server - using kernel ioctls.

https://private-user-images.githubusercontent.com/815726/332149344-c367155a-953c-468c-aa67-261616c17d19.mov

## Why?

Do something cool with a headless server that has HDMI output and barebones CoreOS.
No X or Wayland is required, because it renders directly to the display surface.

When connected to a 4K TV, show something interesting instead of a login console with a tiny font.

This works well on an off-brand digital signage appliance (NUC) with a Celeron CPU N3350 CPU and 4GB RAM, and maxes out at 30 FPS when rendering 4K images
(most likely it's limited by the HDMI interface).

## How to run it

Build and run as a regular Rust project:

```shell
./hypno-toadface [--speed=<speed>] [--sound=<devicepath>] [--no-print-fps]
```

`--speed=<speed>` is an optional argument to specify how fast the animation should be playing, for example `--speed=0.1`. The default speed is 0.04. Negative values make the animation run in reverse.

`--sound=<devicepath>` specifies a path to the ALSA sound device, for example `--sound=/dev/snd/pcmC0D3p`. If not specified, no sound will be played.

`--no-print-fps` turns off printing the FPS counter.

⚠️ This project works without a windowing manager, but in Linux only one device can have exclusive access to the GPU. If X or Wayland is running, using the GPU would be impossible. To run this project, stop any windowing managers.

As accessing displays and audio requires elevated privileges, the safest way to get them is by adding a user to the `video` and `audio` groups:

```shell
# Required for CoreOS with a sparse /etc/group
getent group video >> /etc/group
getent group audio >> /etc/group
usermod -a -G video,audio $USERNAME
```

If SELinux is enabled, make sure that `/etc/group` has the right label by running `ls -Z /etc/group`, it should look something like this:

```
system_u:object_r:passwd_file_t:s0 /etc/group
```

## Run as a systemd service

Create a systemd unit file:

```shell
cat <<EOF > ~/.config/systemd/user/hypno-toadface.service
[Unit]
Description=Hypno Toadface

[Service]
Type=exec
ExecStart=/var/home/core/.local/bin/hypno-toadface --no-print-fps --sound=/dev/snd/pcmC0D3p
KillMode=process
Restart=on-failure

[Install]
WantedBy=default.target
EOF
```

and start it without logging in:

```shell
sudo loginctl enable-linger $USER
```

## References:

* [Vulkan Tutorial](https://vulkan-tutorial.com) - an Excellent tutorial on using Vulkan
* [KMS GLSL](https://github.com/astefanutti/kms-glsl) - a similar project, but using DRM/KMS
* [Shadertoy](https://www.shadertoy.com) - a collection of really impressive shaders
* [Raw ALSA player](https://github.com/PHJArea217/raw-alsa-player) - a pure C example how to play audio without an audioserver
* [Generating pink noise](https://www.firstpr.com.au/dsp/pink-noise/)
* [Colored noise generator](https://mynoise.net/NoiseMachines/whiteNoiseGenerator.php) - a neat noise generator with a lot of fine tuning
* [Biquad calculator](https://www.earlevel.com/main/2021/09/02/biquad-calculator-v3/) - showing a frequency response graph for a biquad filter

ALL GLORY TO THE HYPNOTOAD
