using SeisTimes
using Documenter

DocMeta.setdocmeta!(SeisTimes, :DocTestSetup, :(using SeisTimes); recursive=true)

makedocs(;
    modules=[SeisTimes],
    authors="wtegtow <w.tegtow@gmail.com> and contributors",
    sitename="SeisTimes.jl",
    format=Documenter.HTML(;
        canonical="https://wtegtow.github.io/SeisTimes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/wtegtow/SeisTimes.jl",
    devbranch="main",
)
