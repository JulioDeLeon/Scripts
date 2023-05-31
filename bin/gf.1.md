% gf(1) 0.0.1
% Julio de Leon (jangelod@gmail.com)
% May 2023

# NAME
gf - look for given regex in a file (or files) in a flexible manner

# SYNOPSIS
**gf**  -s *PATTERN* [*OPTION*]

**gf** -s *PATTERN*

# DESCRIPTION
**gf** is a tool which will search for a given *PATTERN* within the filesystem, recursively searching at and below the current working directory. There are options to ignore or target particular files through flags as well.

# OPTION
**-s**, **--search** *PATTERN*
: accepts a pattern which will be searched for within files.

**-t**, **--target** *PATTERN*
: accepts a pattern to search in only files that match the given pattern.

**-i**, **--ignore** *PATTERN*
: accepts a pattern to NOT search in files that match the given pattern.

**-d**, **--debug**
: enable debug logs

**-c**, **--context** *NUMBER*
: number of lines to print to console BEFORE and AFTER finding a match for the given search pattern

**-m** **--maxline** *NUMBER*
: maximum number of lines allowed to read per file. Used to speed up searching and traversal.

# EXAMPLES
**gf -s "myRegex[ab]+"**
: simple example of looking for a pattern recursively within the current working directory.

**gf -s "insomeJson" -t "\\.json"**
: example of searching for a term within files which match a JSON file extension.

**gf -s "insomeJson" --ignore "\\.json"**
: example of searching for a term within files which do not have a JSON file extension.

**gf -s "someTerm" -c 3**
: simple example of looking for a pattern recursively within the current working directory, print 3 lines before and after the line of the matched term within any particular file.

**gf -s "someTerm" -d**
: simple example of looking for a pattern recursively within the current working directory, printing debug logs.

# EXIT VALUES
**0**
: SUCCESS

**1**
: ERROR

# BUGS
Probably
