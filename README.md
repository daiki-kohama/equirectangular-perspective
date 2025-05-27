# equirectangular-perspective

Script to convert equirectangular images to perspective images

## Environment setup

```bash
pip install numpy opencv-python
```

## How to run

```bash
python main.py \
    --input sample_equirectangular.jpg \
    --output output.jpg \
    --output_size 500 500 \
    --v_fov_deg 120 \
    --yaw_deg -90 \
    --pitch_deg -30 \
    --roll_deg 15 \
    --show
```

The coordinate system below use one which OpenCV use.
Please see difference of the coordinate systems [here](https://medium.com/@christophkrautz/what-are-the-coordinates-225f1ec0dd78).

- input: input equirectangular image path
- output: output perspective image path
- output_size: output image size (width, height)
- v_fov_deg: vertical field of view in degree
- yaw_deg: rotation around the y axis
- pitch_deg: rotation around the x axis after yaw rotation
- roll_deg: rotation around the z axis after yaw and pitch rotations
- show: show the output warped image
