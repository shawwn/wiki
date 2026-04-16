#!/usr/bin/env runhaskell
{-# LANGUAGE OverloadedStrings #-}

{- Debian dependencies:
$ sudo apt-get install libghc-hakyll-dev libghc-pandoc-dev libghc-filestore-dev libghc-tagsoup-dev imagemagick s3cmd git
-}

import Codec.Binary.UTF8.String (encode)
import Control.Exception (onException)
import Data.ByteString.Lazy.Char8 (unpack)
import Data.Char (isAlphaNum, isAscii, toLower, isLetter)
import Data.List (isInfixOf, isPrefixOf, nub, sort)
import Data.Maybe (fromMaybe)
import Network.URI (unEscapeString)
import System.IO.Unsafe (unsafePerformIO)
import qualified Data.Map as M (fromList, lookup, Map)
import qualified Data.Text as T
import System.FilePath (takeBaseName, takeExtension)
import Data.FileStore.Utils (runShellCommand)
import Hakyll (applyTemplateList, buildTags, compile, composeRoutes, constField,
               copyFileCompiler, dateField, defaultContext, defaultHakyllReaderOptions, fromFilePath,
               defaultHakyllWriterOptions, fromCapture, getRoute, gsubRoute, hakyll, idRoute, itemIdentifier,
               loadAll, loadAndApplyTemplate, loadBody, makeItem, match, modificationTimeField, mapContext,
               pandocCompilerWithTransformM, relativizeUrls, route, setExtension, pathField, preprocess,
               tagsField, tagsRules, templateCompiler, version, Compiler, Context, Identifier, Item, Pattern, Rules, Tags, unsafeCompiler)
import Hakyll.Web.Redirect (createRedirects)
import System.Exit (ExitCode(ExitFailure))
import Text.HTML.TagSoup (renderTagsOptions,parseTags,renderOptions, optMinimize, Tag(TagOpen))
import Text.Pandoc (nullAttr,
                    Block(Header), HTMLMathMethod(MathML), Inline(..),
                    ObfuscationMethod(NoObfuscation), Pandoc(..), WriterOptions(..))
import Text.Pandoc.Templates (compileTemplate)
import Text.Pandoc.Walk (walk, walkM)
import Text.Printf (printf)
import Data.List.Utils (replace)
-- local custom modules:
import Inflation (nominalToRealInflationAdjuster)
import LinkMetadata (readLinkMetadata, annotateLink, Metadata)
-- redirects are now defined in data files, not as Haskell modules w/constants

import System.Environment (lookupEnv)
ext :: String
ext = unsafePerformIO (fromMaybe "" <$> lookupEnv "EXT")
{-# NOINLINE ext #-}

main :: IO ()
main = hakyll $ do
             preprocess $ print "Redirects parsing..."
             b1 <- readRedirects "static/redirects/Redirects.hs"
             b2 <- readRedirects "static/redirects/Redirects2.hs"
             version "redirects" $ createRedirects $ b1++b2

             let static = route idRoute >> compile copyFileCompiler
             version "static" $ mapM_ (`match` static) [
                                     "docs/**",
                                     "haskell/**.hs",
                                     "images/**",
                                     "**.hs",
                                     "**.sh",
                                     "**.txt",
                                     "**.html",
                                     "**.page",
                                     "**.css",
                                     "static/**",
                                     "static/img/**",
                                     "static/js/**",
                                     "atom.xml",
                                     "index"]
             match "static/templates/*.html" $ compile templateCompiler

             tags <- buildTags "**.page" (fromCapture "tags/*")

             preprocess $ print "Popups parsing..."
             meta <- preprocess readLinkMetadata

             match "**.page" $ do
                 route $ gsubRoute "," (const "") `composeRoutes` gsubRoute "'" (const "") `composeRoutes` gsubRoute " " (const "-") `composeRoutes`
                          setExtension ext
                 let readerOptions = defaultHakyllReaderOptions
                 compile $ do
                     templ <- unsafeCompiler $ do
                         result <- compileTemplate "" ("<div id=\"TOC\">$toc$</div>\n<div id=\"markdownBody\">$body$</div>" :: T.Text)
                         case result of
                             Right t -> return t
                             Left e  -> error e
                     let opts = woptions { writerTemplate = Just templ }
                     pandocCompilerWithTransformM readerOptions opts (unsafeCompiler . pandocTransform meta)
                         >>= loadAndApplyTemplate "static/templates/default.html" (postCtx tags)
                         >>= imgUrls
                         >>= relativizeUrls

             tagsRules tags $ \tag pattern -> do
                 let title = "Tag: " ++ tag
                 route idRoute
                 compile $ tagPage tags title pattern

readRedirects :: FilePath -> Rules [(Identifier, String)]
readRedirects f = do brokenLinks <- preprocess (fmap read (readFile f) :: IO [(FilePath,String)])
                     let brokenLinks' = map (\(a,b) -> (fromFilePath a, b)) brokenLinks
                     return brokenLinks'

woptions :: WriterOptions
woptions = defaultHakyllWriterOptions{ writerSectionDivs = True,
                                       writerTableOfContents = True,
                                       writerColumns = 120,
                                       writerTOCDepth = 4,
                                       writerHTMLMathMethod = MathML,
                                       writerEmailObfuscation = NoObfuscation }

postList :: Tags -> Pattern -> ([Item String] -> Compiler [Item String]) -> Compiler String
postList tags pattern preprocess' = do
    postItemTemplate <- loadBody "static/templates/postitem.html"
    posts' <- loadAll pattern
    posts <- preprocess' posts'
    applyTemplateList postItemTemplate (postCtx tags) posts
tagPage :: Tags -> String -> Pattern -> Compiler (Item String)
tagPage tags title pattern = do
    list <- postList tags pattern (return . id)
    makeItem ""
        >>= loadAndApplyTemplate "static/templates/tags.html"
                (constField "posts" list <> constField "title" title <>
                    defaultContext)
        >>= relativizeUrls

imgUrls :: Item String -> Compiler (Item String)
imgUrls item = do
    rte <- getRoute $ itemIdentifier item
    return $ case rte of
        Nothing -> item
        Just _  -> fmap (unsafePerformIO . addImgDimensions) item

postCtx :: Tags -> Context String
postCtx tags =
    tagsField "tagsHTML" tags <>
    defaultContext <>
    dateField "created" "%d %b %Y" <>
    dateField "modified" "%d %b %Y" <>
    modificationTimeField "modified" "%d %b %Y" <>
    constField "author" "shawwn" <>
    constField "status" "N/A" <>
    constField "confidence" "log" <>
    constField "description" "N/A" <>
    constField "importance" "0" <>
    constField "cssExtension" "" <>
    escapedTitleField "safeURL"
  where escapedTitleField =  mapContext (map toLower . filter isLetter . takeBaseName) . pathField

pandocTransform :: Metadata -> Pandoc -> IO Pandoc
pandocTransform md = walkM (annotateLink md)
                   . walk headerSelflink
                   . walk (nominalToRealInflationAdjuster . convertInterwikiLinks . convertHakyllLinks . addAmazonAffiliate)

addAmazonAffiliate :: Inline -> Inline
addAmazonAffiliate x@(Link _ r (l, t)) =
    if ("amazon.com/" `T.isInfixOf` l) && not ("tag=gwernnet-20" `T.isInfixOf` l)
    then if "?" `T.isInfixOf` l
         then Link nullAttr r (l <> "&tag=gwernnet-20", t)
         else Link nullAttr r (l <> "?tag=gwernnet-20", t)
    else x
addAmazonAffiliate x = x

convertHakyllLinks :: Inline -> Inline
convertHakyllLinks (Link _ ref ("", "")) =
    let ref' = inlinesToURL ref
    in Link nullAttr ref (T.pack ref', T.pack ("Go to wiki page: " ++ ref'))
convertHakyllLinks x = x

inlinesToURL :: [Inline] -> String
inlinesToURL x = let x' = inlinesToString x
                     (a,b) = break (=='%') x'
                 in escape a ++ b

escape :: String -> String
escape = concatMap escapeURIChar
         where escapeURIChar :: Char -> String
               escapeURIChar c | isAscii c && isAlphaNum c = [c]
                               | otherwise                 = concatMap (printf "%%%02X") $ encode [c]

headerSelflink :: Block -> Block
headerSelflink (Header a (href,b,c) d) =
    Header a (href,b,c) [Link nullAttr d ("#" <> href, T.pack ("Link to section: '" ++ inlinesToString d ++ "'"))]
headerSelflink x = x

addImgDimensions :: String -> IO String
addImgDimensions = fmap (renderTagsOptions renderOptions{optMinimize=whitelist}) . mapM staticImg . parseTags
                 where whitelist s = s /= "div" && s /= "script"

staticImg :: Tag String -> IO (Tag String)
staticImg x@(TagOpen "img" xs) = if vector then return x else
                                   do let optimizedH = lookup "height" xs
                                      let optimizedW = lookup "width" xs
                                      case optimizedH of
                                        Just _ -> return x
                                        Nothing -> do case optimizedW of
                                                       Just _ -> return x
                                                       Nothing -> do
                                                                     case path of
                                                                           Nothing -> return x
                                                                           Just p -> do let p' = if head p == '/' then tail p else p
                                                                                        (height,width) <- imageMagick p' `onException` (putStrLn p)
                                                                                        let width' =  show ((read width::Int) `min` 1400)
                                                                                        let height' = show ((read height::Int) `min` 1400)
                                                                                        return (TagOpen "img" (uniq ([("height", height'), ("width", width')]++xs)))
            where uniq = nub . sort
                  path = lookup "src" xs
                  vector = ((takeExtension $ fromMaybe ".png" path) == ".svg") || ("data:image/" `isPrefixOf` (fromMaybe ".png" path))
staticImg x = return x

imageMagick :: FilePath -> IO (String,String)
imageMagick f = do (status,_,bs) <- runShellCommand "./" Nothing "identify" ["-format", "%h %w\n", f]
                   case status of
                     ExitFailure _ -> error f
                     _ -> do let [height, width] = words $ head $ lines $ unpack bs
                             return (height, width)

inlinesToString :: [Inline] -> String
inlinesToString = concatMap go
  where go x = case x of
               Str s    -> T.unpack s
               Emph x'  -> inlinesToString x'
               Code _ s -> T.unpack s
               _        -> " "

convertInterwikiLinks :: Inline -> Inline
convertInterwikiLinks (Link _ ref (interwiki, article)) =
  case T.unpack interwiki of
    ('!':interwiki') ->
        case M.lookup interwiki' interwikiMap of
                Just url  ->
                    let articleS = T.unpack article
                    in case articleS of
                         "" -> Link nullAttr ref (T.pack (url `interwikiurl` inlinesToString ref),
                                                  T.pack (summary $ unEscapeString $ inlinesToString ref))
                         _  -> Link nullAttr ref (T.pack (url `interwikiurl` articleS),
                                                  T.pack (summary articleS))
                Nothing -> Link nullAttr ref (interwiki, article)
            where interwikiurl u a = u ++ (replace "%20" "_" $ replace "%23" "#" $ escape (deunicode a))
                  deunicode = map (\c -> if c == '\8217' then '\'' else c)
                  summary a = interwiki' ++ ": " ++ a
    _ -> Link nullAttr ref (interwiki, article)
convertInterwikiLinks x = x

interwikiMap :: M.Map String String
interwikiMap = M.fromList $ wpInterwikiMap ++ customInterwikiMap
wpInterwikiMap, customInterwikiMap :: [(String, String)]
customInterwikiMap = [("Hackage", "https://hackage.haskell.org/package/"),
                      ("Hawiki", "https://haskell.org/haskellwiki/"),
                      ("Hoogle", "https://www.haskell.org/hoogle/?hoogle=")]
wpInterwikiMap = [("Wikipedia", "https://en.wikipedia.org/wiki/"),
                  ("Wikiquote", "https://en.wikiquote.org/wiki/"),
                  ("Wiktionary", "https://en.wiktionary.org/wiki/")]
