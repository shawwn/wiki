-- TODO:
-- 1. fix Unicode handling: `shellToCommand` seems to mangle Unicode, screwing up abstracts
-- 2. scrape more sites: possibilities include  predictionbook.com, amazon.com, nature.com, longbets.org, *plos.org, ssrn.com, wiley.com, bmj.com, cran.r-project.org, and rand.org
-- 3. bugs in packages: the WMF API omits the need for `-L` in curl but somehow their live demo works anyway (?!); rxvist doesn't appear to support all biorxiv schemas, including the '/early/' links, forcing me to use curl+Tagsoup; the R library 'fulltext' crashes on examples like `ft_abstract(x = c("10.1038/s41588-018-0183-z"))`
-- fix arxiv PDF links

{-# LANGUAGE OverloadedStrings, DeriveGeneric #-}
module LinkMetadata where

import qualified Data.ByteString.Lazy as B (concat, length, unpack, putStrLn)
import qualified Data.ByteString.Lazy.UTF8 as U -- (encode, decode) -- TODO: why doesn't using U.toString fix the Unicode problems?
import Data.Aeson
import GHC.Generics
import Data.List
import Data.Char
import Data.Maybe
import qualified Data.Map.Strict as M -- (fromList, lookup, Map)
import Text.Pandoc
import qualified Data.Text.IO as TIO
import qualified Data.Text as T
import Data.FileStore.Utils (runShellCommand)
import System.Exit (ExitCode(ExitFailure))
import Data.List.Utils
import System.Directory
import System.IO.Temp
import Network.Api.Arxiv hiding (Link)
import Text.HTML.TagSoup -- (renderTagsOptions,parseTags,renderOptions, optMinimize, Tag(TagOpen))
import Text.Show.Pretty (ppShow)

type Metadata = M.Map Path MetadataItem -- (Title, Author, Date, DOI, Abstract)
type MetadataItem = (String, String, String, String, String)
type MetadataList = [(Path, MetadataItem)]
type Path = String

readLinkMetadata :: IO Metadata
readLinkMetadata = do
             -- for hand created definitions, to be saved
             custom <- (fmap (read . T.unpack) $ TIO.readFile "static/metadata/custom.hs") :: IO MetadataList
             -- auto-generated cached definitions; can be deleted if gone stale
             auto   <- (fmap (read . T.unpack) $ TIO.readFile "static/metadata/auto.hs") :: IO MetadataList
             return $ M.union (M.fromList custom) (M.fromList auto) -- left-biased, 'custom' overrides 'auto'

writeLinkMetadata :: Path -> MetadataItem -> IO ()
writeLinkMetadata l m = do auto <- (fmap (read . T.unpack) $ TIO.readFile "static/metadata/auto.hs") :: IO MetadataList
                           let auto' = M.insert l m $ M.fromList auto
                           temp <- writeSystemTempFile "popup-metadata-auto.db.hs" (ppShow $ M.toAscList auto')
                           renameFile temp "static/metadata/auto.hs"
                           -- alternative: because of the slowdown from blind rewriting of the db, it may be easier
                           -- to do one pass, appending everything to a file, and then manually editing file to populate auto.hs:
                           -- appendFile "static/metadata/auto.hs-tmp" (ppShow [(l, m)])

annotateLink :: Metadata -> Inline -> IO Inline
-- Pandoc types: Link = Link Attr [Inline] Target; Attr = (String, [String], [(String, String)]); Target = (String, String)
annotateLink md x@(Link attr text (target, tooltip)) =
  do
     -- normalize: convert 'https://www.shawwn.com/docs/foo.pdf' to '/docs/foo.pdf' and './docs/foo.pdf' to '/docs/foo.pdf'
     -- the leading '/' indicates this is a local shawwn.com file
     let target' = replace "https://www.shawwn.com/" "/" target
     let target'' = if head target' == '.' then drop 1 target' else target'

     let annotated = M.lookup target'' md
     case annotated of
       -- the link has a valid annotation already defined, so build & return
       Just l  -> return $ constructLink x l
       Nothing -> do new <- linkDispatcher target''
                     case new of
                       Nothing -> writeLinkMetadata target'' ("", "", "", "", "") >> return x
                       Just (p,m) -> do
                                       writeLinkMetadata target'' m
                                       return $ constructLink x m
annotateLink _ x = return x

constructLink :: Inline -> MetadataItem -> Inline
constructLink x@(Link _ text (target, tooltip)) (title, author, date, doi, abstract) =
  if abstract == "" then x else -- if no abstract, don't bother
   Link ("", ["docMetadata"],
        (filter (\d -> (snd d) /= "") [("popup-title",title), ("popup-author",author), ("popup-date",date), ("popup-doi",doi), ("popup-abstract",abstract)]))
        text (target, tooltip)
constructLink a b = error $ "Error: a non-Link was passed into 'constructLink'!" ++ show a ++ " " ++ show b

linkDispatcher, wikipedia, gwern, arxiv, biorxiv :: Path -> IO (Maybe (Path, MetadataItem))
linkDispatcher l | "https://en.wikipedia.org/wiki/" `isPrefixOf` l = wikipedia l
                 | "https://arxiv.org/abs/" `isPrefixOf` l = arxiv l
                 | "https://www.biorxiv.org/content/" `isPrefixOf` l = biorxiv l
                 | "https://www.shawwn.com/" `isPrefixOf` l = gwern (drop 22 l)
                 | head l == '/' = gwern (drop 1 l)
                 | otherwise = return Nothing

pdf :: Path -> IO (Maybe (Path, MetadataItem))
pdf p = do (_,_,mb) <- runShellCommand "./" Nothing "exiftool" ["-printFormat", "$Title$/$Author$/$Date$/$DOI", "-Title", "-Author", "-Date", "-DOI", p]
           if B.length mb > 0 then
             do let (etitle:eauthor:edate:edoi:_) = lines $ U.toString mb
                print $ "PDF: " ++ p ++" DOI: " ++ edoi
                aMaybe <- doi2Abstract edoi
                -- if there is no abstract, there's no point in displaying title/author/date since that's already done by tooltip+URL:
                case aMaybe of
                  Nothing -> return Nothing
                  Just a -> return $ Just (p, (trim etitle, trim eauthor, trim edate, edoi, a))
           else return Nothing

-- nested JSON object: eg 'jq .message.abstract'
data Crossref = Crossref { message :: Message } deriving (Show,Generic)
instance FromJSON Crossref
data Message = Message { abstract :: Maybe String } deriving (Show,Generic)
instance FromJSON Message
doi2Abstract :: [Char] -> IO (Maybe String)
doi2Abstract doi = if length doi <7 then return Nothing
                   else do (_,_,bs) <- runShellCommand "./" Nothing "curl" ["--location", "--silent", "https://api.crossref.org/works/"++doi, "--user-agent", "gwern@gwern.net"]
                           if bs=="Resource not found." then return Nothing
                           else let j = eitherDecode bs :: Either String Crossref
                                in case j of -- start unwrapping...
                                    Left e -> error ("Crossref request failed: "++doi++" "++e) >> return Nothing
                                    Right j' -> let j'' = abstract $ message j' in
                                      case j'' of
                                       Nothing -> return Nothing
                                       Just a -> let trimmedAbstract = replace "</jats:p>" "" $ replace "<jats:title>Abstract</jats:title>\n\t  <jats:p>" "" $ trim a
                                                 in return $ Just trimmedAbstract

data WP = WP { title :: !String, extract_html :: !String } deriving (Show,Generic)
instance FromJSON WP
wikipedia p
  | "https://en.wikipedia.org/wiki/Special" `isPrefixOf` p = return Nothing
  | "https://en.wikipedia.org/wiki/User:" `isPrefixOf` p = return Nothing
  | "https://en.wikipedia.org/wiki/Talk:" `isPrefixOf` p = return Nothing
  | "https://en.wikipedia.org/wiki/Category:" `isPrefixOf` p = return Nothing
  | otherwise = do let p' = replace "/" "%2F" $ replace "%20" "_" $ drop 30 p
                   let p'' = [toUpper (head p')] ++ tail p'
                   let p''' = if '#' `elem` p'' then head $ split "#" p'' else p''
                   -- print p''
                   let rq = "https://en.wikipedia.org/api/rest_v1/page/summary/"++p'''++"?redirect=true"
                   -- `--location` is required or redirects will not be followed by *curl*; '?redirect=true' only makes the *API* follow redirects
                   (status,_,bs) <- runShellCommand "./" Nothing "curl" ["--location", "--silent", rq, "--user-agent", "gwern+wikipediascraping@gwern.net"]
                   case status of
                     ExitFailure _ -> putStrLn ("Wikipedia tooltip failed: " ++ p''') >> return Nothing
                     _ -> let j = eitherDecode bs :: Either String WP
                          in case j of
                               Left e -> putStrLn ("WP request failed: " ++ e ++ " " ++ p ++ " " ++ p''') >> return Nothing
                               Right wp -> let wp' = wp in
                                            return $ Just (p, (title wp', "English Wikipedia", "", "", extract_html wp'))

biorxiv p = do (status,_,bs) <- runShellCommand "./" Nothing "curl" ["--location", "--silent", p, "--user-agent", "gwern+biorxivscraping@gwern.net"]
               case status of
                 ExitFailure _ -> putStrLn ("Biorxiv download failed: " ++ p) >> return Nothing
                 _ -> do
                        let b = U.toString bs
                        let f = parseTags b
                        let metas = filter (isTagOpenName "meta") f
                        let title = concatMap (\x@(TagOpen _ (a:b)) -> if snd a == "DC.Title" then snd $ head b else "") metas
                        let date = concatMap (\x@(TagOpen _ (a:b)) -> if snd a == "DC.Date" then snd $ head b else "") metas
                        let author = intercalate ", " $ filter (/="") $ map (\x@(TagOpen _ (a:b)) -> if snd a == "DC.Contributor" then snd $ head b else "") metas
                        let doi = concatMap (\x@(TagOpen _ (a:b)) -> if snd a == "citation_doi" then snd $ head b else "") metas
                        let abstract = replace "<h3>ABSTRACT</h3>" "" $ replace "<h3>Abstract</h3>" "" $ replace "<h3>SUMMARY</h3>" "" $
                                        trim $ concatMap (\x@(TagOpen _ (a:b:c)) ->
                                                             if snd a == "citation_abstract" then snd $ head c else "") metas
                        return $ Just (p, (title, author, date, doi, abstract))


arxiv url = do -- Arxiv direct PDF links are deprecated but sometimes sneak through
               let arxivid = if "/pdf/" `isInfixOf` url && ".pdf" `isSuffixOf` url
                                 then replace "https://arxiv.org/pdf/" "" $ replace ".pdf" "" url
                                 else replace "https://arxiv.org/abs/" "" url
               (status,_,bs) <- runShellCommand "./" Nothing "curl" ["--location","--silent","https://export.arxiv.org/api/query?search_query=id:"++arxivid++"&start=0&max_results=1", "--user-agent", "gwern+arxivscraping@gwern.net"]
               case status of
                 ExitFailure _ -> error ("Filed on Arxiv ID: " ++ arxivid) >> return Nothing
                 _ -> do let tags = parseTags $ U.toString bs
                         let at = getTitle $ drop 8 tags
                         let aau = intercalate ", " $ getAuthorNames tags
                         let ad = take 10 $ getUpdated tags
                         let ado = getDoi tags
                         let aa = trim $ replace "\n" " " $ getSummary tags
                         return $ Just (url, (at, aau, ad, ado, aa))

trim :: String -> String
trim = reverse . dropWhile (isSpace) . reverse . dropWhile (isSpace) . filter (/='\n')

gwern p | ".pdf" `isSuffixOf` p = pdf p
        | otherwise =
            do (status,_,bs) <- runShellCommand "./" Nothing "curl" ["--location", "--silent", "https://www.shawwn.com/"++p, "--user-agent", "shawnpresser+gwernscraping@gmail.com"]
               case status of
                 ExitFailure _ -> putStrLn ("Gwern.net download failed: " ++ p) >> return Nothing
                 _ -> do
                        let b = U.toString bs
                        let f = parseTags b
                        let metas = filter (isTagOpenName "meta") f
                        let title = concatMap (\x@(TagOpen _ (a:b)) -> if snd a == "title" then snd $ head b else "") metas
                        let date = concatMap (\x@(TagOpen _ (a:b)) -> if snd a == "dc.date.issued" then snd $ head b else "") metas
                        let author = concatMap (\x@(TagOpen _ (a:b)) -> if snd a == "author" then snd $ head b else "") metas
                        let doi = ""
                        let abstract      = trim $ renderTags $ filter filterAbstract $ takeWhile takeToAbstract $ dropWhile dropToAbstract f
                        let description = concatMap (\x@(TagOpen _ (a:b)) -> if snd a == "description" then snd $ head b else "") metas
                        -- the description is inferior to the abstract, so we don't want to simply combine them, but if there's no abstract, settle for the description:
                        let abstract'     = if length description > length abstract then description else abstract

                        return $ Just (p, (title, author, author, doi, abstract'))
        where
          dropToAbstract (TagOpen "div" [("id", "abstract")]) = False
          dropToAbstract _ = True
          takeToAbstract (TagClose "div") = False
          takeToAbstract _ = True
          filterAbstract (TagOpen "div" _) = False
          filterAbstract (TagClose "div") = False
          filterAbstract (TagOpen "blockquote" _) = False
          filterAbstract (TagClose "blockquote") = False
          filterAbstract _ = True
