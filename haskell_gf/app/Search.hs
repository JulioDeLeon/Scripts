{-# LANGUAGE OverloadedStrings #-}

module Search where

import Config (ProcessedConfig(..))
import Control.Monad (forM_, when)
import Control.Exception (try, IOException)
import qualified Data.ByteString as B
import qualified Data.Set as Set
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.Directory (doesDirectoryExist, listDirectory)
import System.FilePath ((</>), takeExtension)
import System.IO (IOMode(ReadMode), withFile)
import Text.Regex.TDFA (Regex, matchTest)
import Text.Regex.TDFA.String (regexec)
import System.Console.ANSI

data SearchOptions = SearchOptions
  { patternRegex :: Regex
  , context :: Int
  , maxLine :: Maybe Int
  , debug :: Bool
  }

binaryExtensions :: Set.Set String
binaryExtensions = Set.fromList
  [ ".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o", ".a", ".lib", ".jar", ".war", ".ear", ".class"
  , ".pyc", ".pyo", ".pyd", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".ico", ".svg", ".webp", ".mp4"
  , ".avi", ".mov", ".wmv", ".flv", ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".zip", ".rar", ".7z", ".tar"
  , ".gz", ".bz2", ".xz", ".lzma"
  ]

recursiveSearch :: ProcessedConfig -> SearchOptions -> FilePath -> IO ()
recursiveSearch cfg opts path = do
  isDir <- doesDirectoryExist path
  if isDir
    then do
      contents <- listDirectory path
      forM_ contents $ \name -> do
        let newPath = path </> name
        if not (shouldSkip newPath cfg)
          then recursiveSearch cfg opts newPath
          else when (debug opts) $ putStrLn $ "Skipping " ++ newPath
    else do
      if not (shouldSkip path cfg)
        then searchFile path opts
        else when (debug opts) $ putStrLn $ "Skipping " ++ path

shouldSkip :: FilePath -> ProcessedConfig -> Bool
shouldSkip path cfg =
  let ext = takeExtension path
  in
    any (`matchTest` path) (pIgnores cfg) ||
    (not (null (pTargets cfg)) && not (any (`matchTest` path) (pTargets cfg))) ||
    (Set.member ext binaryExtensions)

isBinaryFile :: FilePath -> IO Bool
isBinaryFile path = do
  content <- withFile path ReadMode $ \h -> B.hGet h 1024
  return $ B.any (== 0) content

searchFile :: FilePath -> SearchOptions -> IO ()
searchFile path opts = do
  isBin <- isBinaryFile path
  if isBin
    then return ()
    else do
      contentEither <- try (TIO.readFile path) :: IO (Either IOException T.Text)
      case contentEither of
        Left _ -> return ()
        Right content -> do
          let linesList = zip [1..] (T.lines content)
          let matchedLines = filter (\(_, line) -> matchTest (patternRegex opts) (T.unpack line)) linesList

          if null matchedLines
            then return ()
            else do
              setSGR [SetColor Foreground Vivid Green]
              putStrLn path
              setSGR [Reset]
              
              mapM_ (printMatch linesList opts) matchedLines
              putStrLn ""

printMatch :: [(Int, T.Text)] -> SearchOptions -> (Int, T.Text) -> IO ()
printMatch allLines opts (lineNum, lineContent) = do
  let ctx = context opts
  let beforeLines = drop (max 0 (lineNum - 1 - ctx)) $ take (lineNum - 1) allLines
  
  mapM_ (\(n, l) -> putStrLn $ "[" ++ show n ++ "]\t" ++ T.unpack l) beforeLines

  putStr $ "[" ++ show lineNum ++ "]\t"
  printHighlighted (T.unpack lineContent) (patternRegex opts)
  putStrLn ""

  let afterLines = take (min (length allLines - lineNum) ctx) $ drop lineNum allLines
  mapM_ (\(n, l) -> putStrLn $ "[" ++ show n ++ "]\t" ++ T.unpack l) afterLines

printHighlighted :: String -> Regex -> IO ()
printHighlighted line regex = do
  let res = regexec regex line
  case res of
    Right (Just (before, match, after, _)) -> do
      putStr before
      setSGR [SetColor Foreground Vivid Red, SetConsoleIntensity BoldIntensity]
      putStr match
      setSGR [Reset]
      printHighlighted after regex
    _ -> putStr line
