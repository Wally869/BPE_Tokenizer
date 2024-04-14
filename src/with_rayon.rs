use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash, time::{Duration, Instant},
};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::Tokenizer;




pub fn parallel_generate_with_base_vocabulary<T>(
    inputs: Vec<Vec<T>>,
    base_vocabulary: Vec<T>,
    target_vocabulary_size: usize,
) -> Tokenizer<T>
where
    T: Eq + Hash + Clone + Debug + Sync,
{
    let mut tokenizer = Tokenizer::default();

    // feed base_vocab to tokenizer
    let mut straight_lookup: HashMap<Vec<T>, usize> = HashMap::new();

    // first pass
    let mut curr_token_value: usize = 0;

    // perform dedup on base input
    {
        let mut token_set = HashSet::new();
        for elem in &base_vocabulary[..] {
            token_set.insert(elem);
        }

        for elem in token_set {
            tokenizer.register(&[elem.to_owned()], curr_token_value);
            straight_lookup.insert(vec![elem.clone()], curr_token_value);
            curr_token_value += 1;
        }
    }

    let now = Instant::now();

    // find data pairs
    while curr_token_value < target_vocabulary_size {
        let rslts: Vec<HashMap<&[T], usize>> = inputs
            .par_iter()
            .map(|curr_input| {
                let mut pairs_count: HashMap<&[T], usize> = HashMap::new();

                let mut pointer: usize = 0;
                while pointer < curr_input.len() {
                    let curr_val_pointer = pointer;

                    tokenizer.tokenize_item_no_write(curr_input, &mut pointer);
                    //pointer += 1;

                    if pointer < curr_input.len() - 1 {
                        //println!("pointer in larger: {}", pointer);
                        pointer += 1;
                        pairs_count
                            .entry(&curr_input[curr_val_pointer..pointer])
                            .and_modify(|count| *count += 1)
                            .or_insert(1);
                    } else {
                        pairs_count
                            .entry(&curr_input[curr_val_pointer..pointer])
                            .and_modify(|count| *count += 1)
                            .or_insert(1);
                    }
                }

                pairs_count
            })
            .collect();

        // consolidate all the pair counts 
        let mut consolidated_pair_counts: HashMap<&[T], usize> = HashMap::new();
        for pc in rslts {
            for (k, v) in pc.iter() {
                consolidated_pair_counts
                .entry(k)
                .and_modify(|count| *count += v)
                .or_insert(*v);
            }
        }

        // find biggest that is not in tokenizer
        let (max_key, _) = consolidated_pair_counts
            .into_iter()
            .filter(|(key, _)| straight_lookup.get(&key.to_owned().to_vec()).is_none())
            .max_by_key(|(_, pair_count)| *pair_count)
            .unwrap();

        // add biggest to tokenizer
        tokenizer.register(max_key, curr_token_value);

        // increment id tracker
        curr_token_value += 1;
    }

    return tokenizer;
}

#[cfg(test)]
mod tests_parallel {
    use crate::test_data::RAW_TEXT;
    use std::{collections::HashSet, fs::File, io::Write};

    use super::parallel_generate_with_base_vocabulary;

    #[test]
    fn test_run_parallel() {
        //let text_val: Vec<char> = RAW_TEXT.chars().collect();
        let subsections: Vec<&str> = RAW_TEXT.split(".").collect();
        println!("Nb Subsections: {}", subsections.len());

        let lens: usize = subsections.iter().map(|s| s.len()).sum::<usize>(); //.collect();
        println!("avg len subsec: {}", lens / subsections.len());

        // to chars
        let subsections_chars: Vec<Vec<char>> = subsections
            .iter()
            .map(|section| section.chars().collect())
            .collect();

        let base_vocab: Vec<char> = {
            let mut set: HashSet<char> = HashSet::new();
            let all_chars: Vec<char> = RAW_TEXT.chars().collect();

            for c in all_chars {
                set.insert(c);
            }

            set.iter().map(|e| *e).collect()
        };

        let tokenizer = parallel_generate_with_base_vocabulary(subsections_chars, base_vocab, 1024);

        let all_chars: Vec<char> = RAW_TEXT.chars().collect();
        let mut token_buffer: Vec<usize> = vec![];
        tokenizer.tokenize(&all_chars, &mut token_buffer, &mut 0);
        println!("tokenized length: {}", token_buffer.len());

        let mut detokenized = vec![];
        tokenizer.detokenize(&token_buffer, &mut detokenized);

        let decoded: String = detokenized.iter().collect();
        assert_eq!(RAW_TEXT, decoded);

        /*
        // save tokenizer to file
        let mut f_tokenizer =
            File::create("./generated/tokenizer.json").expect("Unable to create file");
        f_tokenizer.write(serde_json::to_string(&tokenizer).unwrap().as_bytes());

        // dunk tokens to file
        let mut f = File::create("./generated/generated.bin").expect("Unable to create file");

        for (_, v) in tokenizer.lookup {
            let temp: String = v.iter().collect();
            f.write(temp.as_bytes());
            f.write("\n".as_bytes());
        }
        */
    }
}
