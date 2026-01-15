#!/bin/bash
# Helper script to render OBJ files with Blender
# Usage: ./render_obj.sh <input_obj> [output_png]

INPUT_OBJ="${1}"
OUTPUT_PNG="${2:-render.png}"

if [ -z "$INPUT_OBJ" ]; then
    echo "Usage: $0 <input_obj> [output_png]"
    echo "Example: $0 output/scene_step_10.obj render_step_10.png"
    exit 1
fi

if [ ! -f "$INPUT_OBJ" ]; then
    echo "Error: Input file not found: $INPUT_OBJ"
    exit 1
fi

echo "Rendering $INPUT_OBJ to $OUTPUT_PNG"
blender --background --python render_scene.py -- \
    --input "$INPUT_OBJ" \
    --output "$OUTPUT_PNG" \
    --width 1920 \
    --height 1080 \
    --samples 128 \
    --seed 42

echo "Done! Output saved to $OUTPUT_PNG"
