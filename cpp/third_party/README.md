# Vendored single-header libraries

| file | upstream | version / commit | license |
|---|---|---|---|
| `stb_image_write.h` | https://github.com/nothings/stb (`stb_image_write.h`) | v1.16, commit `2c980bb59875b0d32144a71867fbdebb2f77cd20` (master, committed 2026-08-02), fetched 2026-10-05 | public domain (Unlicense) or MIT, at the user's choice (see the end of the file) |

`stb_image_write.h` SHA-256: `cbd5f0ad7a9cf4468affb36354a1d2338034f2c12473cf1a8e32053cb6914a05`.

The file is unmodified. It is compiled in exactly one translation unit
(`cpp/src/render.cpp`, which defines `STB_IMAGE_WRITE_IMPLEMENTATION` and
`STB_IMAGE_WRITE_STATIC`, so its symbols stay private to that file) and is
used only to encode PNG images in memory (`stbi_write_png_to_func`).

To update: download the file at a pinned commit,

```bash
curl -sSfL https://raw.githubusercontent.com/nothings/stb/<commit>/stb_image_write.h \
     -o cpp/third_party/stb_image_write.h
sha256sum cpp/third_party/stb_image_write.h
```

and record the commit and checksum here.
