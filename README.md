# Hypno Toadface

This is a weekend project, attempting to display some graphics using Linux DRM (Direct Rendering Manager).

## Why?

Do something cool with a headless server that has HDMI output and barebones CoreOS.
No X or Wayland is required, because it renders directly to the display surface.

When connected to a 4K TV, show something interesting instead of a login console with a tiny font.

This works well on an off-brand digital signage appliance (NUC) with a Celeron CPU N3350 CPU and 4GB RAM, and maxes out at 30 FPS when rendering 4K images
(most likely it's limited by the HDMI interface).

## How to run it

Build and run as a regular Rust project:

```shell
./hypno-toadface
```

As accessing displays requires elevated privileges, the safest way to get them is by adding a user to the `video` group:

```shell
# Required for CoreOS with a sparse /etc/group
getent group video >> /etc/group
usermod -a -G video $USERNAME
```

If SELinux is enabled, make sure that `/etc/group` has the right label by running `ls -Z /etc/group`, it should look something like this:

```
system_u:object_r:passwd_file_t:s0 /etc/group
```

## References:

* [Excellent tutorial on using Vulkan](https://vulkan-tutorial.com)
* [A similar project](https://github.com/astefanutti/kms-glsl)
* [Collection of really impressive shaders](https://www.shadertoy.com/)

ALL GLORY TO THE HYPNOTOAD
