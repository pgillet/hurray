// Hurray social preview card, 1280x640. Render with:
//   typst compile card.typ hurray-social-preview.png --ppi 72
#set page(width: 1280pt, height: 640pt, margin: 0pt, fill: rgb("#0b1413"))
#set text(font: "DejaVu Sans", fill: rgb("#e7efed"))

#let ink  = rgb("#f3f8f7")
#let line = rgb("#2a3d39")
#let teal = rgb("#0d9488")
#let mint = rgb("#2dd4bf")
#let mute = rgb("#93a8a4")

#let cell-size = 132pt
#let glyph-size = 70pt   // 34/64 of the cell, matching the SVG wordmark's proportions

// Rotating the h 180deg about its box centre mirrors its baseline, so its two uprights
// end up above the x-height line the other letters sit on. Drop it back down until they
// meet that line and the former ascender reads as a descender. The font metrics put this
// at 0.1455em; typst's line box is not exactly ascent+descent, so the rendered result was
// still 5pt high and this is the measured value, verified in pixels against 'u'.
#let turned-h-drop = 0.217 * glyph-size

#let glyph(c) = align(center + horizon,
  text(font: "DejaVu Sans Mono", size: glyph-size, weight: "bold", fill: ink, c))

#let wordmark = grid(
  columns: (cell-size,) * 6, rows: cell-size, stroke: 2pt + line,
  glyph("h"), glyph("u"), glyph("r"), glyph("r"), glyph("a"),
  grid.cell(fill: teal, align(center + horizon, move(dy: turned-h-drop,
    rotate(180deg, text(font: "DejaVu Sans Mono", size: glyph-size, weight: "bold",
                        fill: white, "h"))))),
)

#align(center + horizon, block(width: 100%, {
  align(center, wordmark)
  v(60pt)
  align(center, text(size: 34pt, fill: mute, "A zero-copy, streamable, language-agnostic"))
  v(14pt)
  align(center, text(size: 34pt, fill: mute, "tensor interchange format"))
  v(46pt)
  align(center, text(size: 26pt, fill: mint, font: "DejaVu Sans Mono",
                     "github.com/pgillet/hurray"))
}))
