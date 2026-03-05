module Main where

import Config (loadConfig, Config(..), compileConfig)
import Search (recursiveSearch, SearchOptions(..))
import Options.Applicative
import qualified Data.Set as Set
import qualified Data.Text as T
import System.Directory (getCurrentDirectory)
import Text.Regex.TDFA (makeRegex, Regex)

data Args = Args
  { argSearch :: String
  , argTarget :: [String]
  , argIgnore :: [String]
  , argContext :: Int
  , argMaxLine :: Maybe Int
  , argDebug :: Bool
  }

argsParser :: Parser Args
argsParser = Args
  <$> strOption
      ( long "search"
     <> short 's'
     <> help "Search pattern" )
  <*> many (strOption
      ( long "target"
     <> short 't'
     <> help "Target files/directories pattern" ))
  <*> many (strOption
      ( long "ignore"
     <> short 'i'
     <> help "Ignore files/directories pattern" ))
  <*> option auto
      ( long "context"
     <> short 'c'
     <> value 0
     <> help "Context lines to show" )
  <*> optional (option auto
      ( long "maxline"
     <> short 'm'
     <> help "Maximum lines to read per file" ))
  <*> switch
      ( long "debug"
     <> help "Enable debug output" )

main :: IO ()
main = do
  args <- execParser opts
  config <- loadConfig
  
  -- Compile all regexes into a single configuration object
  let finalConfig = compileConfig $ config
        { targets = Set.union (targets config) (Set.fromList $ map T.pack $ argTarget args)
        , ignores = Set.union (ignores config) (Set.fromList $ map T.pack $ argIgnore args)
        }

  -- Pre-compile search regex
  let pRegex = makeRegex (argSearch args) :: Regex

  let searchOpts = SearchOptions
        { patternRegex = pRegex
        , context = argContext args
        , maxLine = argMaxLine args
        , debug = argDebug args
        }

  cwd <- getCurrentDirectory
  recursiveSearch finalConfig searchOpts cwd
  where
    opts = info (argsParser <**> helper)
      ( fullDesc
     <> progDesc "Grep File - A fast file searcher"
     <> header "haskell-gf - a Haskell port of gf" )
