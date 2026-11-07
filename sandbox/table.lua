-- Table heads in bold white, for PDFs: the row's fill is in the preamble. Word gets all
-- this from its table style.
function Table(tbl)
  for _, row in ipairs(tbl.head.rows) do
    for _, cell in ipairs(row.cells) do
      for _, block in ipairs(cell.contents) do
        if block.t == "Plain" or block.t == "Para" then
          block.content = pandoc.Inlines({
            pandoc.RawInline("latex", "\\color{white}"), pandoc.Strong(block.content),
          })
        end
      end
    end
  end
  return tbl
end
