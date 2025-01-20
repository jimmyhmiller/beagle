Right now I'm just doing the top level ir
But the way my mapping is working, I have token range to ir range.
But since my compilation unit is per function
I actually have token_range to many different irs
I think I need to make it so that each time I compile a function I make a new mapping.
But I also keep the top level one.
So now, for each file I will have {file_name, function_name, instructions, mapping}