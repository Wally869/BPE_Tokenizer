# BPE Tokenizer  

Generic implementation of a BPE Tokenizer using a tree structure for use in transformers or other AI models.  

## Overview  

I made this tokenizer to tokenize VGM data and train a transformer for music generation. There is an example in lib.rs using a vector of chars, but the method is generic and can be used on any types that implement T: Eq + Hash + Clone + Debug 

Tokenizer is Serializable / Deserializable. See tokenizer.json for a sample generated tokenizer using characters. 

Usage is straightforward with current implementation: 
```rust
/// generate(&Vec<T>, nb_tokens)
let tokenizer = generate(&input, 512);
```

## To Do   
- Improve performance   
Current implementation uses a tree on main thread over a single input array, but we can split the input into multiple smaller inputs, or accept a list as input and split the work over multiple workers using Rayon.  
- Enable length constraint on tokens   


## References  
Merging tokens here  
https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0
