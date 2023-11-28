use std::collections::{HashMap,VecDeque};
#[derive(Clone,Debug,Copy, PartialEq)]
enum Neighbour{
    Reference(usize),
    Value(i32)
}

struct DoubleMap{
    map : HashMap<usize, i32>,
    key_map: HashMap<usize,usize>
}

impl DoubleMap{
    fn new(map: HashMap<usize,i32>)->Self{
        let key_map = map.keys().map(|k| (*k,*k)).collect::<HashMap<usize,usize>>();
        Self { map, key_map }
    }
    fn insert(&mut self, key:usize, val:i32){
        let original_key = key;
        let mut real_key = key; 
        while let Some(&key) = self.key_map.get(&key){
            if key == real_key{
                break;
            }
            real_key = key;
        }
        self.key_map.insert(original_key,real_key); // Shorten the distance for next time...
        self.map.insert(real_key,val);
    }
    fn redirect_key(&mut self, old_key:usize, new_key:usize){
        self.key_map.insert(old_key,new_key);
    }
    fn get(&self,key: &usize)->Option<&i32>{
        let mut real_key = self.get_true_key(*key);
        self.map.get(&real_key)
    }
    fn get_true_key(&self, key:usize) -> usize{
        let mut real_key = key; 
        while let Some(&key) = self.key_map.get(&key){
            if key == real_key{
                break;
            }
            real_key = key;
        }
        real_key
    }
    fn contains_key(&self, key: &usize)->bool{
        let real_key = self.get_true_key(*key);
        self.key_map.contains_key(&real_key)
    }
    fn keys(&self)->Vec<usize>{
        self.key_map.values().map(|v| *v).collect::<Vec<usize>>()
    }
    fn items(&self)->Vec<(usize,i32)>{
        self.key_map.values().map(|v| (*v, *self.map.get(v).unwrap() )).collect::<Vec<(usize,i32)>>()
    }
}
#[derive(Clone)]
struct Matrix{
    data: Vec<i32>,
    rows: usize,
    cols: usize,
}
impl Matrix{
    fn new(nested_vec: Vec<Vec<i32>>)->Self{
        let rows = nested_vec.len();
        let cols = nested_vec[0].len();
        let data = nested_vec.into_iter().map(|v| v).flatten().collect::<Vec<i32>>();
        Self{data,rows,cols}
    }
    fn subtract_walls(mut self, other: &Self)->Self{
        println!("---- Subtract Walls ----");
        self.print();
        let inner_indices = (0..self.data.len()).filter(|i| !self.idx_is_at_edges(*i) ).collect::<Vec<usize>>();
        for i in inner_indices.into_iter(){
            self.data[i] -= other.data[i];
        }
        println!("---- Walls gone ----");
        self.print();
        self
    }
    fn sum(&self)->i32{
        self.data.iter().enumerate().filter(|(i,v)| !self.idx_is_at_edges(*i)).fold(0, |acc,(_,v)| acc + v)
    }
    fn print(&self){
        for i in 0..self.rows{
            println!("{:?}", &self.data[(i*self.cols)..((i+1)*self.cols)]);
        }
        println!();
    }
    fn idx_to_coord(&self,idx:usize)->[usize;2]{
        let [x,y] = [idx / self.cols, idx % self.cols];
        [x,y]
    }
    fn coord_to_idx(&self, coord:[usize;2])->usize{
        coord[0] * self.cols + coord[1]
    }
    fn idx_is_at_edges(&self, idx:usize)->bool{
        // println!("Max idx: {}", self.data.len()-1);
        let [x,y] = self.idx_to_coord(idx);
        // println!("idx: {idx} -> ({x},{y})");
        if x == 0 || x == self.rows-1{
            true
        }else if y == 0 || y == self.cols-1{
            true
        }else{
            false
        }
    }
    fn get_edge_indices(&self) -> Vec<usize>{
        (0..self.data.len()).filter(|i| self.idx_is_at_edges(*i) ).collect::<Vec<usize>>()
    }
    fn get_neighbour_idx(&self, idx:usize)->Vec<usize>{
        // Messy. Should be refactored.
        let [x,y] = [ (idx / self.cols) as i128, (idx % self.cols) as i128 ];
        let combinations = [ [x+1,y], [x-1,y], [x,y+1], [x,y-1] ];
        let neighbours = combinations
            .iter()
            .filter_map(|[x,y]|{
                if x < &0i128 || y < &0i128 || x > &(usize::MAX as i128) || y > &(usize::MAX as i128){
                    None
                }else{
                    let idx_raw = x*(self.cols as i128) + y;
                    if idx_raw > usize::MAX as i128{
                        None
                    }else{
                        let idx = idx_raw as usize;
                        if idx >= self.data.len(){
                            None
                        }else{
                            Some(idx as usize)
                        }
                    }
                }
            }).collect::<Vec<usize>>();
        neighbours
    }
    fn adjust_heights_for_leaks(mut self)->Self{
        let mut known = DoubleMap::new(HashMap::new());
        let mut to_visit = VecDeque::<usize>::new();
        // Define all edges as known.
        // Define all keys to have their initial value.
        // Define all non_edge neighbours of edges to be points to visit.
        (0..self.data.len()).for_each(|i| { 
            if self.idx_is_at_edges(i){
                known.insert(i, self.data[i]);
            }else{
                to_visit.push_back(i);
                known.redirect_key(i, i);

            }

        });

        while let Some(idx) = to_visit.pop_front(){
            let neighbours = self.get_neighbour_idx(idx);
            let undefined_neighbours = neighbours.iter().filter(|i| !known.contains_key(i)).map(|i|*i).collect::<Vec<usize>>();
            let undefined_high_neighbours = undefined_neighbours.iter().filter(|i| self.data[**i] >= self.data[idx]).map(|i| *i).collect::<Vec<usize>>();
            if undefined_high_neighbours.is_empty(){
                let min_neigbhour = neighbours.iter().fold(None,|acc: Option<&usize>,i: &usize|{
                    if let Some(&acc_idx) = acc{
                        if self.data[acc_idx] > self.data[*i]{
                            Some(i)
                        }else{
                            acc
                        }
                    }else{
                        Some(i)
                    }
                });
                if let Some(v) =  min_neigbhour{
                    self.data[idx] = std::cmp::max(self.data[idx], self.data[*v]);
                    known.insert(idx, self.data[idx]);
                }
            }else{
                // Deal with some neighbours not being defined:
                
            }
        }


        self

    }
}

fn main(){
    todo!()
}