using CSV
using Dates
using HTTP

"""
    dt2unix(dt)

Transforms `dt` to unix datetime format.
"""
dt2unix(dt) = round(Int, datetime2unix(DateTime(dt)))

"""
    get_yahoo_data(
        ticker::String, dt_start, dt_end; 
        cols=["Open", "High", "Low", "Close", "Volume"]
    )

Fetches data from Yahoo Finance for the given ticker and time period.
"""
function get_yahoo_data(
    ticker::String, dt_from=nothing, dt_to=nothing;
    cols=["Open", "High", "Low", "Close", "Volume"]
)
    dt_from = isnothing(dt_from) ? "1900-01-01" : dt_from
    dt_to = isnothing(dt_to) ? now() : dt_to
    url = string(
        "https://query1.finance.yahoo.com/v7/finance/download/", ticker, 
        "?period1=", dt2unix(dt_from), "&period2=", dt2unix(dt_to), 
        "&interval=1d&events=history&includeAdjustedClose=true")
    data = CSV.read(HTTP.get(url).body, NamedTuple)
    Float32.(hcat([data[Symbol(c)] for c ∈ cols]...))
end