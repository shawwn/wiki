{- LinkMetadata.hs: module for generating Pandoc links which are annotated with metadata, which can then be displayed to the user as 'popups' by /static/js/popups.js. These popups can be excerpts, abstracts, article introductions etc, and make life much more pleasant for the reader - hover over link, popup, read, decide whether to go to link.
Author: Gwern Branwen
Date: 2019-08-20
License: CC-0
-}

{-# LANGUAGE OverloadedStrings, DeriveGeneric #-}
module LinkMetadata where

import qualified Data.ByteString.Lazy as B (length)
import qualified Data.ByteString.Lazy.UTF8 as U (toString)
import Data.Aeson
import GHC.Generics
import Data.List
import Data.Char
import Data.Maybe
import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Text.Pandoc
import Data.FileStore.Utils (runShellCommand)
import System.Exit (ExitCode(ExitFailure))
import Data.List.Utils (replace, split)
import System.Directory
import System.IO.Temp
import Text.HTML.TagSoup
import Text.Show.Pretty (ppShow)

type Metadata = M.Map Path MetadataItem -- (Title, Author, Date, DOI, Abstract)
type MetadataItem = (String, String, String, String, String)
type MetadataList = [(Path, MetadataItem)]
type Path = String

readLinkMetadata :: IO Metadata
readLinkMetadata = do
             custom <- (fmap (read . T.unpack) $ TIO.readFile "static/metadata/custom.hs") :: IO MetadataList
             auto   <- (fmap (read . T.unpack) $ TIO.readFile "static/metadata/auto.hs") :: IO MetadataList
             return $ M.union (M.fromList custom) (M.fromList auto)

writeLinkMetadata :: Path -> MetadataItem -> IO ()
writeLinkMetadata l m = do auto <- (fmap (read . T.unpack) $ TIO.readFile "static/metadata/auto.hs") :: IO MetadataList
                           let auto' = M.insert l m $ M.fromList auto
                           temp <- writeSystemTempFile "popup-metadata-auto.db.hs" (ppShow $ M.toAscList auto')
                           renameFile temp "static/metadata/auto.hs"

annotateLink :: Metadata -> Inline -> IO Inline
annotateLink md x@(Link attr text (target, tooltip)) =
  do
     let targetS  = T.unpack target
     let target'  = replace "https://www.shawwn.com/" "/" targetS
     let target'' = if null target' then target' else if head target' == '.' then drop 1 target' else target'

     let annotated = M.lookup target'' md
     print (attr, text, target, tooltip, annotated)
     case annotated of
       Just l  -> return $ constructLink x l
       Nothing -> do new <- linkDispatcher target''
                     case new of
                       Nothing    -> writeLinkMetadata target'' ("", "", "", "", "") >> return x
                       Just (_,m) -> writeLinkMetadata target'' m >> return (constructLink x m)
annotateLink _ x = return x

constructLink :: Inline -> MetadataItem -> Inline
constructLink x@(Link _ text (target, tooltip)) (title, author, date, doi, abstract) =
  if abstract == "" then x else
   Link ("", ["docMetadata"],
        filter (\d -> snd d /= "")
          [ ("popup-title",    T.pack title)
          , ("popup-author",   T.pack author)
          , ("popup-date",     T.pack date)
          , ("popup-doi",      T.pack doi)
          , ("popup-abstract", T.pack abstract) ])
        text (target, tooltip)
constructLink a b = error $ "Error: a non-Link was passed into 'constructLink'! This should never happen." ++ show a ++ " " ++ show b

linkDispatcher, wikipedia, gwern, arxiv, biorxiv :: Path -> IO (Maybe (Path, MetadataItem))
linkDispatcher l | "https://en.wikipedia.org/wiki/" `isPrefixOf` l = wikipedia l
                 | "https://arxiv.org/abs/"         `isPrefixOf` l = arxiv l
                 | "https://www.biorxiv.org/content/" `isPrefixOf` l = biorxiv l
                 | "https://www.shawwn.com/"        `isPrefixOf` l = gwern (drop 22 l)
                 | not (null l), head l == '/'                      = gwern (drop 1 l)
                 | otherwise                                         = return Nothing

pdf :: Path -> IO (Maybe (Path, MetadataItem))
pdf p = do (_,_,mb) <- runShellCommand "./" Nothing "exiftool" ["-printFormat", "$Title$/$Author$/$Date$/$DOI", "-Title", "-Author", "-Date", "-DOI", p]
           if B.length mb > 0 then
             do let (etitle:eauthor:edate:edoi:_) = lines $ U.toString mb
                print $ "PDF: " ++ p ++" DOI: " ++ edoi
                aMaybe <- doi2Abstract edoi
                case aMaybe of
                  Nothing -> return Nothing
                  Just a  -> return $ Just (p, (trim etitle, trim eauthor, trim edate, edoi, a))
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
                                in case j of
                                    Left e  -> putStrLn ("Error: Crossref request failed: "++doi++" "++e) >> return Nothing
                                    Right j' -> let j'' = abstract $ message j' in
                                      case j'' of
                                       Nothing -> return Nothing
                                       Just a  -> let trimmedAbstract = replace "</jats:p>" "" $ replace "<jats:title>Abstract</jats:title>\n\t  <jats:p>" "" $ trim a
                                                  in return $ Just trimmedAbstract

data WP = WP { title :: !String, extract_html :: !String } deriving (Show,Generic)
instance FromJSON WP
wikipedia p
  | "https://en.wikipedia.org/wiki/Special"  `isPrefixOf` p = return Nothing
  | "https://en.wikipedia.org/wiki/User:"    `isPrefixOf` p = return Nothing
  | "https://en.wikipedia.org/wiki/Talk:"    `isPrefixOf` p = return Nothing
  | "https://en.wikipedia.org/wiki/Category:" `isPrefixOf` p = return Nothing
  | otherwise = do let p'   = replace "/" "%2F" $ replace "%20" "_" $ drop 30 p
                   let p''  = [toUpper (head p')] ++ tail p'
                   let p''' = if '#' `elem` p'' then head $ split "#" p'' else p''
                   let rq   = "https://en.wikipedia.org/api/rest_v1/page/summary/"++p'''++"?redirect=true"
                   (status,_,bs) <- runShellCommand "./" Nothing "curl" ["--location", "--silent", rq, "--user-agent", "gwern+wikipediascraping@gwern.net"]
                   case status of
                     ExitFailure _ -> putStrLn ("Wikipedia tooltip failed: " ++ p''') >> return Nothing
                     _ -> let j = eitherDecode bs :: Either String WP
                          in case j of
                               Left e   -> putStrLn ("WP request failed: " ++ e ++ " " ++ p ++ " " ++ p''') >> return Nothing
                               Right wp -> return $ Just (p, (title wp, "English Wikipedia", "", "", extract_html wp))

biorxiv p = do (status,_,bs) <- runShellCommand "./" Nothing "curl" ["--location", "--silent", p, "--user-agent", "gwern+biorxivscraping@gwern.net"]
               case status of
                 ExitFailure _ -> putStrLn ("Biorxiv download failed: " ++ p) >> return Nothing
                 _ -> do
                        let b     = U.toString bs
                        let f     = parseTags b
                        let metas = filter (isTagOpenName "meta") f
                        let titleV   = concatMap (\(TagOpen _ (a:b')) -> if snd a == "DC.Title"       then snd $ head b' else "") metas
                        let dateV    = concatMap (\(TagOpen _ (a:b')) -> if snd a == "DC.Date"        then snd $ head b' else "") metas
                        let authorV  = intercalate ", " $ filter (/="") $ map (\(TagOpen _ (a:b')) -> if snd a == "DC.Contributor" then snd $ head b' else "") metas
                        let doiV     = concatMap (\(TagOpen _ (a:b')) -> if snd a == "citation_doi"  then snd $ head b' else "") metas
                        let abstractV = replace "<h3>ABSTRACT</h3>" "" $ replace "<h3>Abstract</h3>" "" $ replace "<h3>SUMMARY</h3>" "" $
                                        trim $ concatMap (\(TagOpen _ (a:_:c)) ->
                                                             if snd a == "citation_abstract" then snd $ head c else "") metas
                        return $ Just (p, (titleV, authorV, dateV, doiV, abstractV))


arxiv url = do let arxivid = if "/pdf/" `isInfixOf` url && ".pdf" `isSuffixOf` url
                                 then replace "https://arxiv.org/pdf/" "" $ replace ".pdf" "" url
                                 else replace "https://arxiv.org/abs/" "" url
               (status,_,bs) <- runShellCommand "./" Nothing "curl" ["--location","--silent","https://export.arxiv.org/api/query?search_query=id:"++arxivid++"&start=0&max_results=1", "--user-agent", "gwern+arxivscraping@gwern.net"]
               case status of
                 ExitFailure _ -> putStrLn ("Error: on Arxiv ID " ++ arxivid) >> return Nothing
                 _ -> do let tags = parseTags $ U.toString bs
                         let at  = getTitle  $ drop 8 tags
                         let aau = intercalate ", " $ getAuthorNames tags
                         let ad  = take 10 $ getUpdated tags
                         let ado = getDoi tags
                         let aa  = trim $ replace "\n" " " $ getSummary tags
                         return $ Just (url, (at, aau, ad, ado, aa))

trim :: String -> String
trim = reverse . dropWhile isSpace . reverse . dropWhile isSpace . filter (/='\n')

gwern p | ".pdf" `isSuffixOf` p = pdf p
        | otherwise =
            do (status,_,bs) <- runShellCommand "./" Nothing "curl" ["--location", "--silent", "https://www.shawwn.com/"++p, "--user-agent", "shawnpresser+gwernscraping@gmail.com"]
               case status of
                 ExitFailure _ -> putStrLn ("Gwern.net download failed: " ++ p) >> return Nothing
                 _ -> do
                        let b     = U.toString bs
                        let f     = parseTags b
                        let metas = filter (isTagOpenName "meta") f
                        let titleV  = concatMap (\(TagOpen _ (a:b')) -> if snd a == "title"           then snd $ head b' else "") metas
                        let dateV   = concatMap (\(TagOpen _ (a:b')) -> if snd a == "dc.date.issued"  then snd $ head b' else "") metas
                        let authorV = concatMap (\(TagOpen _ (a:b')) -> if snd a == "author"          then snd $ head b' else "") metas
                        let doi     = ""
                        let abstractV     = trim $ renderTags $ filter filterAbstract $ takeWhile takeToAbstract $ dropWhile dropToAbstract f
                        let description   = concatMap (\(TagOpen _ (a:b')) -> if snd a == "description" then snd $ head b' else "") metas
                        let abstract'     = if length description > length abstractV then description else abstractV

                        return $ Just (p, (titleV, authorV, dateV, doi, abstract'))
        where
          dropToAbstract (TagOpen "div" [("id", "abstract")]) = False
          dropToAbstract _                                     = True
          takeToAbstract (TagClose "div") = False
          takeToAbstract _               = True
          filterAbstract (TagOpen "div" _)    = False
          filterAbstract (TagClose "div")     = False
          filterAbstract (TagOpen "blockquote" _) = False
          filterAbstract (TagClose "blockquote")  = False
          filterAbstract _                    = True

inlinesToString :: [Inline] -> String
inlinesToString = concatMap go
  where go x = case x of
               Str s    -> T.unpack s
               Code _ s -> T.unpack s
               _        -> " "

-- ---------------------------------------------------------------------------
-- Arxiv Atom XML helpers (previously from Network.Api.Arxiv)
-- ---------------------------------------------------------------------------

getTagTextContent :: String -> [Tag String] -> String
getTagTextContent name ts = case dropWhile (not . isTagOpenName name) ts of
  (_:TagText s:_) -> s
  _               -> ""

getTitle :: [Tag String] -> String
getTitle = getTagTextContent "title"

getSummary :: [Tag String] -> String
getSummary = getTagTextContent "summary"

getUpdated :: [Tag String] -> String
getUpdated = getTagTextContent "updated"

getDoi :: [Tag String] -> String
getDoi = getTagTextContent "arxiv:doi"

getAuthorNames :: [Tag String] -> [String]
getAuthorNames [] = []
getAuthorNames ts =
  case dropWhile (not . isTagOpenName "author") ts of
    []       -> []
    (_:rest) ->
      let name      = getTagTextContent "name" rest
          remaining = drop 1 $ dropWhile notAuthorClose rest
      in if null name then [] else name : getAuthorNames remaining
  where notAuthorClose (TagClose "author") = False
        notAuthorClose _                   = True
