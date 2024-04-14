use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash, time::Instant,
};

use serde::{Deserialize, Serialize};

pub mod with_rayon;

#[cfg(test)]
pub mod test_data;


#[derive(Serialize, Deserialize, Debug)]
pub struct Node<T>
where
    T: Eq + Hash + Clone + Debug,
{
    pub byte_value: T,
    /// we are guaranteed that subpairs exist, this node definitely has a value
    /// so no need for options
    pub token_value: usize,
    pub children: HashMap<T, Node<T>>,
}

impl<T> Node<T>
where
    T: Eq + Hash + Clone + Debug,
{
    pub fn new(byte_value: &[T], token_value: usize) -> Node<T> {
        if byte_value.len() == 1 {
            return Node {
                byte_value: byte_value[0].to_owned(),
                token_value: token_value,
                children: HashMap::new(),
            };
        } else {
            let mut children = HashMap::new();
            children.insert(
                byte_value[1].to_owned(),
                Node::new(&byte_value[1..], token_value),
            );

            return Node {
                byte_value: byte_value[0].to_owned(),
                token_value: 0,
                children: children,
            };
        }
    }

    pub fn register(&mut self, byte_value: &[T], token_value: usize) {
        if byte_value.len() == 1 {
            // unordered tokens, update value here
            self.token_value = token_value;
        } else {
            match self.children.get_mut(&byte_value[1]) {
                None => {
                    let child_node = Node::new(&byte_value[1..], token_value);
                    self.children.insert(byte_value[1].to_owned(), child_node);
                }
                Some(child) => {
                    child.register(&byte_value[1..], token_value);
                }
            }
        }
    }

    pub fn tokenize(
        &self,
        read_buffer: &Vec<T>,
        write_buffer: &mut Vec<usize>,
        pointer: &mut usize,
    ) {
        *pointer += 1;

        // check if not oob
        if *pointer < read_buffer.len() {
            match self.children.get(&read_buffer[*pointer]) {
                Some(child) => child.tokenize(read_buffer, write_buffer, pointer),
                None => write_buffer.push(self.token_value),
            }
        } else {
            write_buffer.push(self.token_value);
        }
    }

    fn tokenize_no_write(&self, read_buffer: &[T], pointer: &mut usize) {
        *pointer += 1;

        if *pointer < read_buffer.len() {
            match self.children.get(&read_buffer[*pointer]) {
                None => (),
                Some(child) => child.tokenize_no_write(read_buffer, pointer),
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Tokenizer<T>
where
    T: Eq + Hash + Clone + Debug,
{
    pub children: HashMap<T, Node<T>>,
    pub lookup: HashMap<usize, Vec<T>>,
}

impl<T> Tokenizer<T>
where
    T: Eq + Hash + Clone + Debug,
{
    pub fn default() -> Self {
        return Tokenizer {
            children: HashMap::new(),
            lookup: HashMap::new(),
        };
    }

    pub fn register(&mut self, token: &[T], token_value: usize) {
        self.lookup.insert(token_value, token.to_vec().to_owned());

        match self.children.get_mut(&token[0]) {
            None => {
                let child = Node::new(token, token_value);
                self.children.insert(token[0].to_owned(), child);
                self.lookup.insert(token_value, token.to_vec().to_owned());
            }
            Some(child) => {
                child.register(token, token_value);
            }
        }
    }

    pub fn tokenize(
        &self,
        read_buffer: &Vec<T>,
        write_buffer: &mut Vec<usize>,
        pointer: &mut usize,
    ) {
        while *pointer < read_buffer.len() {
            match self.children.get(&read_buffer[*pointer]) {
                None => panic!("no child in tokenizer that matches"),
                Some(child) => child.tokenize(read_buffer, write_buffer, pointer),
            }
        }
    }

    /// From buffer find token and move pointer for single element
    fn tokenize_item_no_write(&self, buffer: &[T], pointer: &mut usize) {
        match self.children.get(&buffer[*pointer]) {
            None => panic!("child not found"),
            Some(child) => {
                child.tokenize_no_write(buffer, pointer);
            }
        }
    }

    pub fn detokenize(&self, read_buffer: &Vec<usize>, write_buffer: &mut Vec<T>) {
        for elem in read_buffer {
            write_buffer.extend_from_slice(self.lookup.get(elem).unwrap());
        }
    }
}

pub fn generate<T>(input: &Vec<T>, target_vocabulary_size: usize) -> Tokenizer<T>
where
    T: Eq + Hash + Clone + Debug,
{
    let mut tokenizer = Tokenizer::default();

    let mut straight_lookup: HashMap<Vec<T>, usize> = HashMap::new();

    // first pass
    let mut curr_token_value: usize = 0;

    // perform dedup on base input
    {
        let mut token_set = HashSet::new();
        for elem in &input[..] {
            token_set.insert(elem);
        }

        for elem in token_set {
            tokenizer.register(&[elem.to_owned()], curr_token_value);
            straight_lookup.insert(vec![elem.clone()], curr_token_value);
            curr_token_value += 1;
        }
    }


    let now = Instant::now();
    // now create pairs
    while curr_token_value < target_vocabulary_size {
        // create pairs by using the tokenizer to tokenize input values
        let mut pairs_count: HashMap<&[T], usize> = HashMap::new();

        let mut pointer: usize = 0;
        while pointer < input.len() {
            let curr_val_pointer = pointer;

            tokenizer.tokenize_item_no_write(input, &mut pointer);
            //pointer += 1;

            if pointer < input.len() - 1 {
                //println!("pointer in larger: {}", pointer);
                pointer += 1;
                pairs_count
                    .entry(&input[curr_val_pointer..pointer])
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            } else {
                pairs_count
                    .entry(&input[curr_val_pointer..pointer])
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }

        // find biggest that is not in tokenizer
        let (max_key, _) = pairs_count
            .into_iter()
            .filter(|(key, _)| straight_lookup.get(&key.to_owned().to_vec()).is_none())
            .max_by_key(|(_, pair_count)| *pair_count)
            .unwrap();

        // add biggest to tokenizer
        tokenizer.register(max_key, curr_token_value);

        // increment id tracker
        curr_token_value += 1;
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);


    return tokenizer;
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Write};

    use super::generate;

    use super::test_data::RAW_TEXT;

    #[test]
    fn try_it_out() {
        let text_val: Vec<char> = RAW_TEXT.chars().collect();
        println!("text len: {}", text_val.len());

        let tokenizer = generate(&text_val, 1024);

        let mut token_buffer: Vec<usize> = vec![];
        tokenizer.tokenize(&text_val, &mut token_buffer, &mut 0);
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
