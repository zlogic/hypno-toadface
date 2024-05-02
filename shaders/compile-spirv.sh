#!/bin/sh
rm -f *.spv
for SHADER_FILE in *.glsl; do
    SHADER_NAME=$(basename $SHADER_FILE | sed s/\.glsl\$//)
    echo "Compiling $ENTRYPOINT_NAME..."
    glslang -V $SHADER_FILE -o ${SHADER_NAME}.spv
done
