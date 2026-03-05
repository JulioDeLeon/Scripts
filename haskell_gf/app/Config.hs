{-# LANGUAGE OverloadedStrings #-}

module Config where

import qualified Data.Set as Set
import System.Directory (doesFileExist, getHomeDirectory)
import System.FilePath ((</>))
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Control.Monad (foldM)
import Text.Regex.TDFA (Regex, makeRegex)

data Config = Config
  { ignores :: Set.Set T.Text
  , targets :: Set.Set T.Text
  } deriving (Show, Eq)

data ProcessedConfig = ProcessedConfig
  { pIgnores :: [Regex]
  , pTargets :: [Regex]
  }

compileConfig :: Config -> ProcessedConfig
compileConfig cfg = ProcessedConfig
  { pIgnores = map (makeRegex . T.unpack) (Set.toList $ ignores cfg)
  , pTargets = map (makeRegex . T.unpack) (Set.toList $ targets cfg)
  }

defaultConfig :: Config
defaultConfig = Config Set.empty Set.empty

loadConfig :: IO Config
loadConfig = do
  let systemConfig = "/etc/gfconf"
  home <- getHomeDirectory
  let userConfig = home </> ".gfconf"

  cfg1 <- parseFile defaultConfig systemConfig
  cfg2 <- parseFile cfg1 userConfig
  return cfg2

parseFile :: Config -> FilePath -> IO Config
parseFile cfg path = do
  exists <- doesFileExist path
  if not exists
    then return cfg
    else do
      content <- TIO.readFile path
      foldM parseLine cfg (T.lines content)

parseLine :: Config -> T.Text -> IO Config
parseLine cfg line
  | T.null trimmed || T.isPrefixOf "#" trimmed = return cfg
  | "source " `T.isPrefixOf` trimmed = do
      let sourcePath = T.unpack $ T.strip $ T.drop 7 trimmed
      parseFile cfg sourcePath
  | "target " `T.isPrefixOf` trimmed = do
      let target = T.strip $ T.drop 7 trimmed
      return $ cfg { targets = Set.insert target (targets cfg) }
  | "ignore " `T.isPrefixOf` trimmed = do
      let ignore = T.strip $ T.drop 7 trimmed
      return $ cfg { ignores = Set.insert ignore (ignores cfg) }
  | otherwise = return cfg
  where
    trimmed = T.strip line
