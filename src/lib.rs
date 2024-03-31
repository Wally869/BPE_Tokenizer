use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
};

use serde::{Deserialize, Serialize};

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

    return tokenizer;
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Write};

    use super::generate;

    #[test]
    fn try_it_out() {
        let raw_text = r#"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Arcu odio ut sem nulla pharetra diam. Turpis tincidunt id aliquet risus feugiat in ante metus. Ipsum dolor sit amet consectetur adipiscing. Neque sodales ut etiam sit amet nisl purus in. Tincidunt nunc pulvinar sapien et ligula. Feugiat nisl pretium fusce id velit ut tortor pretium. Odio ut sem nulla pharetra diam sit amet nisl suscipit. Commodo quis imperdiet massa tincidunt nunc pulvinar sapien. Lectus magna fringilla urna porttitor rhoncus dolor purus non. Mi proin sed libero enim sed faucibus turpis in eu. Elementum sagittis vitae et leo duis ut diam quam.

        Fusce id velit ut tortor pretium. Sagittis vitae et leo duis ut diam. Scelerisque eu ultrices vitae auctor. Nullam vehicula ipsum a arcu cursus vitae. Pretium nibh ipsum consequat nisl. Fringilla ut morbi tincidunt augue. Etiam dignissim diam quis enim. Viverra aliquet eget sit amet tellus. Neque aliquam vestibulum morbi blandit cursus. Aliquam sem fringilla ut morbi tincidunt augue. Mauris cursus mattis molestie a iaculis at erat. Nisi quis eleifend quam adipiscing vitae proin sagittis. Ut tortor pretium viverra suspendisse potenti nullam ac tortor. At augue eget arcu dictum varius duis at consectetur.
        
        Faucibus in ornare quam viverra orci. Nec ultrices dui sapien eget mi. Mattis enim ut tellus elementum sagittis. Consectetur lorem donec massa sapien faucibus et molestie. Cursus eget nunc scelerisque viverra mauris in. Eu sem integer vitae justo. Vel orci porta non pulvinar. Ac turpis egestas sed tempus urna et pharetra pharetra. Commodo sed egestas egestas fringilla phasellus faucibus. Senectus et netus et malesuada fames ac turpis. Amet mauris commodo quis imperdiet massa tincidunt nunc pulvinar sapien. A scelerisque purus semper eget duis. Elementum facilisis leo vel fringilla est ullamcorper eget nulla. Gravida dictum fusce ut placerat orci nulla pellentesque dignissim enim. Mi bibendum neque egestas congue quisque egestas diam in. Aliquam etiam erat velit scelerisque in dictum non consectetur a.
        
        Donec massa sapien faucibus et molestie ac feugiat sed. Aliquam purus sit amet luctus venenatis lectus magna fringilla. Sit amet consectetur adipiscing elit pellentesque. Tincidunt augue interdum velit euismod in pellentesque massa placerat. Phasellus vestibulum lorem sed risus ultricies tristique nulla aliquet. Urna neque viverra justo nec ultrices dui sapien eget. Pellentesque elit ullamcorper dignissim cras tincidunt. Pharetra convallis posuere morbi leo. Feugiat in ante metus dictum at tempor commodo ullamcorper a. Non curabitur gravida arcu ac tortor dignissim convallis. Rhoncus mattis rhoncus urna neque viverra justo.
        
        Sagittis orci a scelerisque purus semper eget duis. Cursus vitae congue mauris rhoncus. Cras fermentum odio eu feugiat pretium nibh ipsum consequat. Morbi leo urna molestie at elementum. Viverra vitae congue eu consequat ac felis donec et odio. Nibh sed pulvinar proin gravida hendrerit. Amet commodo nulla facilisi nullam. Viverra aliquet eget sit amet. Vitae tortor condimentum lacinia quis vel eros donec. Nec ultrices dui sapien eget mi proin. Cursus vitae congue mauris rhoncus. Nullam vehicula ipsum a arcu cursus vitae congue mauris rhoncus. Vestibulum rhoncus est pellentesque elit ullamcorper. Neque convallis a cras semper auctor. Urna duis convallis convallis tellus id interdum velit laoreet. Convallis a cras semper auctor neque vitae tempus quam. Quis blandit turpis cursus in.
        
        Accumsan in nisl nisi scelerisque eu ultrices. Ornare suspendisse sed nisi lacus. Facilisis mauris sit amet massa vitae tortor. Velit ut tortor pretium viverra suspendisse potenti nullam ac tortor. Fusce ut placerat orci nulla pellentesque. At tempor commodo ullamcorper a lacus. Nec feugiat nisl pretium fusce id. Imperdiet dui accumsan sit amet nulla facilisi morbi. Tellus rutrum tellus pellentesque eu. Quis enim lobortis scelerisque fermentum dui.
        
        Nisl vel pretium lectus quam id. Amet mauris commodo quis imperdiet massa. Urna id volutpat lacus laoreet non curabitur gravida arcu ac. Nulla posuere sollicitudin aliquam ultrices sagittis orci a scelerisque. Aenean euismod elementum nisi quis eleifend. Egestas dui id ornare arcu. Lectus nulla at volutpat diam ut venenatis tellus. Odio aenean sed adipiscing diam donec adipiscing. Commodo odio aenean sed adipiscing diam donec adipiscing tristique. Et tortor consequat id porta. Sed euismod nisi porta lorem mollis aliquam ut. Id faucibus nisl tincidunt eget nullam non nisi est. Fringilla ut morbi tincidunt augue interdum velit euismod in pellentesque. Amet luctus venenatis lectus magna fringilla. Leo duis ut diam quam nulla porttitor massa. Morbi tincidunt ornare massa eget egestas purus viverra accumsan.
        
        Quam elementum pulvinar etiam non quam lacus suspendisse. Viverra adipiscing at in tellus. Ut consequat semper viverra nam libero justo laoreet sit. Nibh tortor id aliquet lectus proin nibh nisl condimentum id. Id faucibus nisl tincidunt eget nullam non nisi est sit. Mattis nunc sed blandit libero volutpat sed. Nam aliquam sem et tortor consequat id porta. Pellentesque habitant morbi tristique senectus et netus et malesuada. Nunc scelerisque viverra mauris in aliquam sem. Vitae elementum curabitur vitae nunc sed velit dignissim sodales. Pellentesque id nibh tortor id aliquet lectus. Sollicitudin aliquam ultrices sagittis orci a scelerisque purus semper. Ridiculus mus mauris vitae ultricies leo integer. Dignissim enim sit amet venenatis urna.
        
        Enim lobortis scelerisque fermentum dui faucibus in ornare. Aliquet lectus proin nibh nisl condimentum id venenatis. In tellus integer feugiat scelerisque. Pulvinar pellentesque habitant morbi tristique senectus et netus et malesuada. Justo eget magna fermentum iaculis eu non diam phasellus. Odio pellentesque diam volutpat commodo sed egestas egestas. Tincidunt eget nullam non nisi est sit amet facilisis magna. Quis blandit turpis cursus in hac habitasse. Porttitor lacus luctus accumsan tortor posuere ac. Fames ac turpis egestas maecenas pharetra convallis posuere morbi leo. Sollicitudin ac orci phasellus egestas tellus rutrum. Odio ut enim blandit volutpat maecenas volutpat blandit aliquam. Malesuada fames ac turpis egestas. Urna condimentum mattis pellentesque id nibh tortor. Lacus luctus accumsan tortor posuere. Nisl vel pretium lectus quam id leo in vitae turpis. Velit egestas dui id ornare arcu odio ut sem. Tincidunt tortor aliquam nulla facilisi cras fermentum odio eu. Quis enim lobortis scelerisque fermentum dui faucibus. Nascetur ridiculus mus mauris vitae.
        
        Risus sed vulputate odio ut enim. Vel orci porta non pulvinar neque laoreet suspendisse. Proin sagittis nisl rhoncus mattis rhoncus urna neque viverra justo. Dui ut ornare lectus sit amet est placerat in egestas. Quis auctor elit sed vulputate mi. Sit amet volutpat consequat mauris nunc congue nisi. Eu augue ut lectus arcu bibendum. Tellus rutrum tellus pellentesque eu tincidunt tortor aliquam. Convallis convallis tellus id interdum velit laoreet. Suspendisse interdum consectetur libero id faucibus nisl. Tellus in metus vulputate eu. Malesuada pellentesque elit eget gravida cum sociis natoque penatibus et. Velit laoreet id donec ultrices tincidunt arcu non sodales neque. Eget lorem dolor sed viverra ipsum nunc aliquet.
        
        Gravida neque convallis a cras. Consequat nisl vel pretium lectus. Massa id neque aliquam vestibulum morbi blandit cursus risus at. Sagittis nisl rhoncus mattis rhoncus urna. Justo nec ultrices dui sapien eget mi. Consectetur adipiscing elit ut aliquam purus sit amet luctus. Amet facilisis magna etiam tempor orci eu. Diam quis enim lobortis scelerisque fermentum dui faucibus in. Nam aliquam sem et tortor consequat id porta. Enim lobortis scelerisque fermentum dui faucibus in ornare quam viverra. Scelerisque in dictum non consectetur a erat nam. Aliquam eleifend mi in nulla. Amet justo donec enim diam. Nunc lobortis mattis aliquam faucibus. Nunc sed blandit libero volutpat sed cras ornare arcu dui. Ut ornare lectus sit amet est placerat in egestas erat.
        
        In pellentesque massa placerat duis ultricies lacus sed. Nibh ipsum consequat nisl vel pretium lectus quam id. Leo urna molestie at elementum eu facilisis sed odio morbi. Vel facilisis volutpat est velit egestas dui id ornare. Vel facilisis volutpat est velit egestas dui id ornare. Est ullamcorper eget nulla facilisi etiam. Vitae aliquet nec ullamcorper sit amet. Urna et pharetra pharetra massa massa ultricies mi quis hendrerit. Cras tincidunt lobortis feugiat vivamus at augue eget. Quam lacus suspendisse faucibus interdum posuere lorem ipsum dolor. Velit laoreet id donec ultrices tincidunt arcu non sodales neque. Scelerisque fermentum dui faucibus in ornare quam. Commodo quis imperdiet massa tincidunt nunc pulvinar sapien et. In egestas erat imperdiet sed. Laoreet id donec ultrices tincidunt. Porta non pulvinar neque laoreet suspendisse interdum consectetur.
        
        At risus viverra adipiscing at. Neque gravida in fermentum et sollicitudin ac orci. Suspendisse in est ante in. Elementum pulvinar etiam non quam lacus. Et sollicitudin ac orci phasellus egestas. Est ante in nibh mauris cursus. Sed arcu non odio euismod lacinia at quis risus sed. Maecenas sed enim ut sem viverra. Bibendum est ultricies integer quis auctor elit sed vulputate. Elementum nisi quis eleifend quam. Eu lobortis elementum nibh tellus molestie nunc non blandit massa. Tellus mauris a diam maecenas sed enim ut sem. Consequat id porta nibh venenatis cras sed felis eget velit. Mauris commodo quis imperdiet massa tincidunt nunc. Blandit massa enim nec dui nunc mattis enim ut. Viverra adipiscing at in tellus integer feugiat scelerisque varius. Risus at ultrices mi tempus imperdiet nulla malesuada.
        
        Montes nascetur ridiculus mus mauris vitae. Elit duis tristique sollicitudin nibh sit amet. Adipiscing elit pellentesque habitant morbi. Cursus vitae congue mauris rhoncus aenean. Luctus accumsan tortor posuere ac ut. Ridiculus mus mauris vitae ultricies leo integer. Habitasse platea dictumst quisque sagittis purus sit amet volutpat. Eu augue ut lectus arcu bibendum at. Pellentesque massa placerat duis ultricies lacus. Pulvinar etiam non quam lacus. Ultricies lacus sed turpis tincidunt id.
        
        Nunc faucibus a pellentesque sit amet porttitor. Ligula ullamcorper malesuada proin libero. In ante metus dictum at tempor commodo ullamcorper a lacus. Risus quis varius quam quisque id diam vel. Amet tellus cras adipiscing enim eu turpis egestas. Dui ut ornare lectus sit amet est placerat. Nisl tincidunt eget nullam non nisi. Tincidunt lobortis feugiat vivamus at augue eget arcu dictum varius. Senectus et netus et malesuada fames ac turpis. Aliquam sem et tortor consequat id porta. Amet consectetur adipiscing elit pellentesque habitant morbi tristique senectus. Hac habitasse platea dictumst vestibulum rhoncus est pellentesque elit ullamcorper. Ut pharetra sit amet aliquam id diam. Quis viverra nibh cras pulvinar mattis nunc sed blandit libero. Sit amet justo donec enim diam vulputate. Adipiscing tristique risus nec feugiat. Ut placerat orci nulla pellentesque dignissim enim.
        
        Turpis tincidunt id aliquet risus feugiat in. Mus mauris vitae ultricies leo integer malesuada. Tellus molestie nunc non blandit massa enim nec. Malesuada proin libero nunc consequat interdum varius. Lacinia quis vel eros donec ac odio tempor. Quis auctor elit sed vulputate mi sit amet mauris. Tincidunt id aliquet risus feugiat in ante metus. Vitae justo eget magna fermentum iaculis eu non. Massa enim nec dui nunc mattis. Quam nulla porttitor massa id neque aliquam vestibulum morbi blandit. Et malesuada fames ac turpis egestas sed tempus urna et. Scelerisque viverra mauris in aliquam. Enim nulla aliquet porttitor lacus luctus accumsan tortor posuere ac.
        
        Condimentum mattis pellentesque id nibh tortor id. Rhoncus est pellentesque elit ullamcorper dignissim cras tincidunt. Eget velit aliquet sagittis id consectetur purus ut. Dignissim diam quis enim lobortis scelerisque fermentum dui faucibus. Orci sagittis eu volutpat odio facilisis mauris sit. Aliquam ut porttitor leo a diam. Sed vulputate odio ut enim blandit. Consectetur a erat nam at lectus urna duis convallis convallis. Ut tortor pretium viverra suspendisse potenti nullam ac tortor vitae. Dui faucibus in ornare quam viverra orci. Fusce ut placerat orci nulla pellentesque dignissim. Imperdiet proin fermentum leo vel orci porta non. Lorem dolor sed viverra ipsum nunc aliquet bibendum. Elementum tempus egestas sed sed risus. Porttitor lacus luctus accumsan tortor posuere ac. Ut venenatis tellus in metus vulputate. Dignissim diam quis enim lobortis scelerisque. Pellentesque habitant morbi tristique senectus et netus et. Turpis massa tincidunt dui ut ornare lectus sit amet. Malesuada proin libero nunc consequat interdum varius sit amet.
        
        Non tellus orci ac auctor. Augue ut lectus arcu bibendum at varius vel pharetra vel. Porttitor rhoncus dolor purus non enim praesent. Tortor consequat id porta nibh venenatis cras. Egestas maecenas pharetra convallis posuere morbi leo urna. Ullamcorper sit amet risus nullam eget felis eget nunc. Aliquam purus sit amet luctus venenatis lectus magna. Massa eget egestas purus viverra accumsan in nisl. Eu feugiat pretium nibh ipsum. Viverra nam libero justo laoreet. Tellus cras adipiscing enim eu turpis egestas pretium aenean. Justo laoreet sit amet cursus sit. Pharetra magna ac placerat vestibulum lectus mauris ultrices eros in. Augue eget arcu dictum varius duis at consectetur lorem. Pulvinar elementum integer enim neque volutpat.
        
        Quam elementum pulvinar etiam non. Porttitor leo a diam sollicitudin tempor. Magna eget est lorem ipsum. Quis varius quam quisque id diam vel quam elementum. Nisi quis eleifend quam adipiscing. Venenatis lectus magna fringilla urna porttitor rhoncus dolor purus non. Gravida dictum fusce ut placerat orci nulla. Lacus luctus accumsan tortor posuere ac ut consequat semper viverra. Magna eget est lorem ipsum dolor. Amet aliquam id diam maecenas. Lacus viverra vitae congue eu. Dui accumsan sit amet nulla facilisi morbi tempus iaculis.
        
        Arcu cursus euismod quis viverra nibh cras pulvinar mattis nunc. Massa placerat duis ultricies lacus sed turpis tincidunt id. Molestie nunc non blandit massa enim nec dui nunc mattis. Scelerisque varius morbi enim nunc faucibus a pellentesque sit amet. Amet risus nullam eget felis eget nunc lobortis. Enim sit amet venenatis urna. Faucibus in ornare quam viverra orci sagittis eu volutpat odio. Adipiscing tristique risus nec feugiat in. Pellentesque habitant morbi tristique senectus et netus et malesuada. Nullam vehicula ipsum a arcu cursus vitae congue mauris. Quam viverra orci sagittis eu volutpat odio. In tellus integer feugiat scelerisque varius morbi enim nunc faucibus. Diam vel quam elementum pulvinar. At lectus urna duis convallis convallis tellus id interdum velit. Id ornare arcu odio ut sem nulla. Fames ac turpis egestas maecenas pharetra convallis posuere. Massa placerat duis ultricies lacus. Elit sed vulputate mi sit amet. Suspendisse faucibus interdum posuere lorem.
        
        In ante metus dictum at tempor commodo. Nunc aliquet bibendum enim facilisis gravida neque convallis a cras. Elit scelerisque mauris pellentesque pulvinar pellentesque habitant morbi tristique senectus. Ac turpis egestas sed tempus urna et pharetra pharetra. Amet nisl suscipit adipiscing bibendum. Ut morbi tincidunt augue interdum velit euismod. Congue quisque egestas diam in arcu cursus euismod quis. Amet cursus sit amet dictum sit. Pellentesque habitant morbi tristique senectus et netus. Amet cursus sit amet dictum sit amet justo donec. Lectus sit amet est placerat. Lacus suspendisse faucibus interdum posuere lorem ipsum dolor. At volutpat diam ut venenatis tellus in metus vulputate. Sit amet porttitor eget dolor morbi non arcu risus. Risus nullam eget felis eget nunc lobortis. Adipiscing bibendum est ultricies integer quis auctor elit sed. Pellentesque eu tincidunt tortor aliquam nulla. Mi eget mauris pharetra et ultrices neque ornare aenean euismod. Pretium vulputate sapien nec sagittis aliquam malesuada.
        
        Turpis massa sed elementum tempus egestas sed sed risus pretium. Nunc pulvinar sapien et ligula. Interdum velit euismod in pellentesque massa placerat duis ultricies. Enim sit amet venenatis urna cursus eget nunc scelerisque. Nisl pretium fusce id velit ut tortor pretium. Lobortis mattis aliquam faucibus purus in massa tempor. Lectus nulla at volutpat diam ut venenatis. Sit amet dictum sit amet justo donec. In eu mi bibendum neque egestas congue quisque. Lacinia at quis risus sed vulputate odio ut enim blandit. Lectus nulla at volutpat diam ut. Penatibus et magnis dis parturient montes. Condimentum lacinia quis vel eros donec.
        
        Sit amet consectetur adipiscing elit duis tristique sollicitudin nibh. Consectetur lorem donec massa sapien. Tincidunt augue interdum velit euismod in pellentesque massa. Vitae tortor condimentum lacinia quis vel eros donec ac. Scelerisque eleifend donec pretium vulputate sapien nec sagittis aliquam. In ornare quam viverra orci sagittis eu volutpat odio facilisis. Sed vulputate odio ut enim blandit volutpat maecenas. Duis convallis convallis tellus id interdum. Elit scelerisque mauris pellentesque pulvinar pellentesque habitant. Ut aliquam purus sit amet luctus venenatis lectus. Suspendisse ultrices gravida dictum fusce ut placerat orci nulla. Ut tortor pretium viverra suspendisse potenti nullam ac tortor vitae. Aliquam eleifend mi in nulla posuere sollicitudin aliquam ultrices sagittis. Aliquam malesuada bibendum arcu vitae elementum curabitur vitae nunc.
        
        Amet consectetur adipiscing elit pellentesque habitant morbi tristique. Facilisi etiam dignissim diam quis enim lobortis. Magna ac placerat vestibulum lectus mauris ultrices eros in. Malesuada pellentesque elit eget gravida cum sociis natoque penatibus et. Aliquam faucibus purus in massa tempor nec feugiat. Lectus quam id leo in. Egestas quis ipsum suspendisse ultrices gravida dictum fusce ut. Volutpat commodo sed egestas egestas fringilla. Donec ultrices tincidunt arcu non sodales neque. Amet volutpat consequat mauris nunc congue. Sagittis purus sit amet volutpat consequat mauris."#;

        let text_val: Vec<char> = raw_text.chars().collect();
        println!("text len: {}", text_val.len());

        let tokenizer = generate(&text_val, 512);

        let mut token_buffer: Vec<usize> = vec![];
        tokenizer.tokenize(&text_val, &mut token_buffer, &mut 0);
        println!("tokenized length: {}", token_buffer.len());

        let mut detokenized = vec![];
        tokenizer.detokenize(&token_buffer, &mut detokenized);

        let decoded: String = detokenized.iter().collect();
        assert_eq!(raw_text, decoded);

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
    }
}