# Generative In-between Line-drawing Interpolation

## Environment

```bash
conda create -n LineDiff python=3.13
conda activate LineDiff
pip install -r requirements.txt
```

## Weights

You can download the pre-trained weights via this [URL](https://www.alipan.com/s/wqtDwMUMqaH). This is a self-extracting archive file made by [`makeself`](https://github.com/megastep/makeself). Download `weights.run` file into the project directory and run:

```bash
./weights.run --keep
```

You can get rid of `weights.run` file after unzipping the `weights` folder.

<details>
    <summary>Not working?</summary>
Try `chmod u+x weights.run` to give the file the right permission.
</details>

## Test

Run:

```bash
python test.py --frame1 <path to the first frame> --frame2 <path to the second frame> --output <path to the output file>
```

## References

* [FlowDiffuser](https://github.com/LA30/FlowDiffuser)
* [GMFSS](https://github.com/98mxr/GMFSS_Fortuna)
* [AnimeRun](https://github.com/lisiyao21/AnimeRun)

