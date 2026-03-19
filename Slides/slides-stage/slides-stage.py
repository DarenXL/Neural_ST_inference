import typst

typst.compile("main.typ", output="test.pdf", font_paths=["Inconsolata/"])

# import touying
# 
# touying.to_html("main.typ", font_paths=["Inconsolata/"])