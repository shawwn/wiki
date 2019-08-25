{-# LANGUAGE OverloadedStrings #-}
module Inflation (nominalToRealInflationAdjuster) where

import Text.Pandoc
import Text.Printf (printf, PrintfArg)
import Data.List (intercalate, unfoldr)

-- Experimental module for implementing automatic inflation adjustment of nominal date-stamped dollar amounts to provide real prices; this is particularly critical in any economics or technology discussion where a nominal price from 1950 is 11x a 2019 real price!

{- Examples:
Markdown~>HTML:

'[$50.50]($1970)'
~>
'<span class="inflationAdjusted" data-originalYear="1970" data-originalAmount="50.50" data-currentYear="2019" data-currentAmount="343.83">$50.50<sub>1970</sub><sup>$343.83</sup></span>'

Testbed:

$ echo '[$50.50]($1970)' | pandoc -w native
[Para [Link ("",[],[]) [Str "$50.50"] ("$1970","")]]

> nominalToRealInflationAdjuster $ Link ("",[],[]) [Str "$50.50"] ("$1970","")
Span ("",["inflationAdjusted"],[("originalYear","1970"),("originalAmount","50.50"),("currentYear","2019"),("currentAmount","343.83")]) [Str "$50.50",Subscript [Str "1970"],Superscript [Str "$343.83"]]

$ echo 'Span ("",["inflationAdjusted"],[("originalYear","1970"),("originalAmount","50.50"),("currentYear","2019"),("currentAmount","343.83")]) [Str "$50.50",Subscript [Str "1970"],Superscript [Str "$343.83"]]' | pandoc -f native -w html
<span class="inflationAdjusted" data-originalYear="1970" data-originalAmount="50.50" data-currentYear="2019" data-currentAmount="343.83">$50.50<sub>1970</sub><sup>$343.83</sup></span>
-}

minPercentage :: Float
minPercentage = 1 + 0.15
currentYear :: Int
currentYear = 2019

nominalToRealInflationAdjuster :: Inline -> Inline
nominalToRealInflationAdjuster (Link _ text (target, _))
  | head target == '$' = if (adjustedDollar / oldDollar) < minPercentage
                        then Str ("$"++oldDollarString) -- if the adjustment is <15%, don't bother, it's not misleading enough yet to need adjusting
                        else Span ("",
                                   ["inflationAdjusted"],
                                   -- provide all 4 variables as metadata the <span> tags for CSS or JS processing
                                   [("originalYear",oldYear),("originalAmount",oldDollarString),
                                    ("currentYear",show currentYear),("currentAmount",adjustedDollarString)])
                             -- [Str ("$" ++ oldDollarString), Subscript [Str oldYear, Superscript [Str ("$"++adjustedDollarString)]]]
                             [Str ("$"++oldDollarString), Math InlineMath ("_{"++oldYear++"}^{\\$"++adjustedDollarString++"}")]

    where oldYear = tail target -- '$1970' ~> '1970'
          oldDollarString = filter (/= '$') $ inlinesToString text -- '$50.50' ~> '50.50'
          oldDollar = read (filter (/=',') oldDollarString) :: Float
          -- control potentially spurious precision:
          -- round to 2 digits when converting to String if a decimal was present and the inflation factor is <10x, otherwise, round to whole numbers.
          -- So, '$1.05' becomes '$20.55', but '$1' becomes '$20' instead of '$20.2359002', and '$0.05' can still become '$0.97'
          precision = if ('.' `elem` oldDollarString) && ((adjustedDollar < 10*oldDollar) || (adjustedDollar < 1)) then "2" else "0"
          adjustedDollar = dollarAdjust oldDollar oldYear
          adjustedDollarString = formatDecimal adjustedDollar precision
nominalToRealInflationAdjuster x = x

inlinesToString :: [Inline] -> String
inlinesToString = concatMap go
  where go x = case x of
               Str s    -> s
               Code _ s -> s
               _        -> " "

-- dollarAdjust "5.50" "1950" ~> "59.84"
dollarAdjust :: Float -> String -> Float
dollarAdjust amount year = let oldYear = read year :: Int in inflationAdjustUS amount oldYear currentYear

-- http://www.usinflationcalculator.com/inflation/consumer-price-index-and-annual-percent-changes-from-1913-to-2008/
-- 0th: 1913 ... 104th: 2017; repeat last inflation rate indefinitely to project forward for 2018+
inflationRatesUS :: [Float]
inflationRatesUS = [0.0,1.0,2.0,12.6,18.1,20.4,14.5,2.6,-10.8,-2.3,2.4,0.0,3.5,-1.1,-2.3,-1.2,0.6,-6.4,-9.3,-10.3,0.8,1.5,3.0,1.4,2.9,-2.8,0.0,0.7,9.9,9.0,3.0,2.3,2.2,18.1,8.8,3.0,-2.1,5.9,6.0,0.8,0.7,-0.7,0.4,3.0,2.9,1.8,1.7,1.4,0.7,1.3,1.6,1.0,1.9,3.5,3.0,4.7,6.2,5.6,3.3,3.4,8.7,12.3,6.9,4.9,6.7,9.0,13.3,12.5,8.9,3.8,3.8,3.9,3.8,1.1,4.4,4.4,4.6,6.1,3.1,2.9,2.7,2.7,2.5,3.3,1.7,1.6,2.7,3.4,1.6,2.4,1.9,3.3,3.4,2.5,4.1,0.1,2.7,1.5,3.0,1.7,1.5,0.8,0.7,2.1,2.1] ++ repeat 2.1

-- inflationAdjustUS 1 1950 2019 ~> 10.88084
-- inflationAdjustUS 5.50 1950 2019 ~> 59.84462
inflationAdjustUS :: Float -> Int -> Int -> Float
inflationAdjustUS d yOld yCurrent = if yOld>=1913 && yCurrent>=1913 then d * totalFactor else d
  where slice from to xs = take (to - from + 1) (drop from xs)
        percents = slice (yOld-1913) (yCurrent-1913) inflationRatesUS
        rates = map (\r -> 1 + (r/100)) percents
        totalFactor = product rates


-- prettyprint decimals with commas for generating larger amounts like "$50,000"
-- https://stackoverflow.com/a/4408556
formatDecimal :: (Ord a, Fractional a, Text.Printf.PrintfArg a) => a -> String -> String
formatDecimal d prec
    | d < 0.0   = "-" ++ formatPositiveDecimal (-d)
    | otherwise = formatPositiveDecimal d
    where formatPositiveDecimal = uncurry (++) . mapFst addCommas . span (/= '.') . printf ("%0."++ prec ++ "f")
          addCommas = reverse . intercalate "," . unfoldr splitIntoBlocksOfThree . reverse
          splitIntoBlocksOfThree l = case splitAt 3 l of ([], _) -> Nothing; p-> Just p
          -- https://hackage.haskell.org/package/fgl-5.7.0.1/docs/src/Data.Graph.Inductive.Query.Monad.html#mapFst
          mapFst :: (a -> b) -> (a, c) -> (b, c)
          mapFst f (x,y) = (f x,y)
