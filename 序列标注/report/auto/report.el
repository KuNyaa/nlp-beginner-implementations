(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "nips13submit_e"
    "times"
    "hyperref"
    "url"
    "xeCJK")
   (TeX-add-symbols
    "fix"
    "new")))

