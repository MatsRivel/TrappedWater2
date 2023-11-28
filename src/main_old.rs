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
    fn get_edge_neighbours(&self,idx:usize)->Vec<usize>{
        self.get_neighbour_idx(idx).iter().filter(|i| self.idx_is_at_edges(**i)).map(|i| *i).collect::<Vec<usize>>()
    }
    fn get_inner_neighbours(&self,idx:usize)->Vec<usize>{
        self.get_neighbour_idx(idx).iter().filter(|i| !self.idx_is_at_edges(**i)).map(|i| *i).collect::<Vec<usize>>()
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
    fn get_min_neighbour(neighbours: &Vec<Neighbour>)->Option<i32>{
        neighbours
            .iter()
            .fold(None, |acc,v|{
                if let Neighbour::Value(height) = v{
                    if acc.is_none() || *height < acc.unwrap() {
                        Some(*height)
                    }else{
                        acc
                    }
                }else{ // We know every neighbour is defined, not just a reference to another one.
                    panic!("This is impossible, we've already checked.")
                }
            })
    }
    fn all_neighbours_are_defined(neighbours: &Vec<Neighbour>)->bool{
        neighbours
        .iter()
        .fold(true,|acc,v|{
            if let Neighbour::Value(_) = v{
                acc && true
            }else{
                false
            }
        })
    }
    fn get_candidates_from_neighbours(&self, idx:usize, known: &DoubleMap)->Vec<Neighbour>{
        self.get_neighbour_idx(idx)
            .iter()
            .filter(|n| self.data[**n] >= self.data[idx]) // Only care about adjacent walls taht are HIGHER than our own.
            .map(|n|{
                if let Some(v) = known.get(n){
                    Neighbour::Value(*v) // We know the value of this neighbour.
                }else{
                    Neighbour::Reference(*n) // We just know which neighbour to check later.
                }
            }).collect::<Vec<Neighbour>>()
    }
    fn detect_two_way_reference(&self, idx:usize, neighbours: &Vec<Neighbour>, known: &DoubleMap) -> Option<Neighbour>{
        let ref_neigh = neighbours
            .iter()
            .filter_map(|v| {
                match v{
                    Neighbour::Reference(_) => Some(*v),
                    Neighbour::Value(_) => None,
                }
            });
        for n in ref_neigh{
            if let Neighbour::Reference(ref_idx) = n{
                for m in self.get_candidates_from_neighbours(ref_idx, known).iter(){
                    if let Neighbour::Reference(potential_idx) = m{
                        if *potential_idx == idx{
                            return Some(n);
                        }
                    }
                }
            }
        }
        None
    }
    fn find_smallest_wall(mut self)->Self{
        // Try not collecting this.
        println!("-- Initial Reference Map ---");
        self.print();
        let mut known: DoubleMap = DoubleMap::new(HashMap::new());
        let mut neighbour_references: HashMap<usize,Vec<Neighbour>> = HashMap::new();
        (0..self.data.len())
            .filter(|i: &usize| self.idx_is_at_edges(*i) )
            .for_each(|i| {
                known.insert(i,self.data[i]);
                neighbour_references.insert(i, vec![Neighbour::Value(self.data[i])]);
            });
        let mut indices = (0..self.data.len())
            .filter(|i| !self.idx_is_at_edges(*i))
            .collect::<VecDeque<usize>>();
        #[cfg(test)]
        let mut counter = 0;
        // We pop_front so that we finish each outer layer before moving inwards once.
        while let Some(idx) = indices.pop_front(){ 
            #[cfg(test)]{
                counter +=1;
                if counter > 1000{
                    panic!("Ran for too long!")
                }
            }
            // No edge found, get edge from neighbours!
            let mut neighbours: Vec<Neighbour> = self.get_candidates_from_neighbours(idx,&known);
            // println!("Idx: {idx}, neighbours: {neighbours:?}");
            // Check if two neighbours are both waiting for each other.
            let two_way_reference = self.detect_two_way_reference(idx, &neighbours, &known);
            // If so, "merge" them, and remove one from the current's neighbours. 
            if let Some(Neighbour::Reference(partner)) = two_way_reference{
                println!("Two way reference between {idx} and {partner}");
                known.redirect_key(idx, partner);
                if let Some(remove_idx) = neighbours.iter().position(|n| *n == Neighbour::Reference(partner)){
                    neighbours.remove(remove_idx);
                }
            }
            // println!("Idx: {idx}, neighbours: {neighbours:?}");
            // If we find an edge, record it and move on.
            if !known.contains_key(&idx){
                println!("_ idx: {idx}, n: {neighbours:?}");
                if Matrix::all_neighbours_are_defined(&neighbours){
                    let min_neighbour = Matrix::get_min_neighbour(&neighbours);
                    if let Some(min_n) = min_neighbour{
                        let new_height = std::cmp::max(self.data[idx],min_n );
                        self.data[idx] = new_height;
                    }
                    known.insert(idx,self.data[idx]);
                }else{
                    println!("| idx: {idx}, n: {neighbours:?}");
                    // Still undefined neighbours. 
                    neighbours.iter().for_each(|n|{
                        if let Neighbour::Reference(other_idx) = n{
                            indices.push_back(*other_idx);
                        }
                    });
                    let adjusted_idx = known.get_true_key(idx);
                    indices.push_back(adjusted_idx); // Put itself back into the rotation until conflicts are resolved.
                }
            }
            #[cfg(test)]{
                if idx == 5 || idx == 10{
                    println!("Idx: {idx}, Known: {:?}, nr: {:?}",known.get(&idx), neighbour_references.get(&idx));
                }
            }
        }
        let to_replaces = known.items().iter().filter_map(|(key,val)| {
            if !self.idx_is_at_edges(*key){
                Some((*key,*val))
            }else{
                None
            }
        } ).collect::<Vec<(usize,i32)>>();
        for (key,val) in to_replaces.into_iter(){
            self.data[key] = val;
        }
        println!("-- Final Reference Map ---");
        self.print();
        self
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
}
pub fn trap_rain_water(height_map: Vec<Vec<i32>>) -> i32 {
    let map = Matrix::new(height_map);
    let surrounding_wall_height = map.clone().find_smallest_wall();
    let volume_trapped_water = surrounding_wall_height.subtract_walls(&map).sum();
    volume_trapped_water

}
fn main(){
    let height_map = vec![vec![1,4,3,1,3,2],vec![3,2,1,3,2,4],vec![2,3,3,2,3,1]];
    assert_eq!(trap_rain_water(height_map),4)
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn min_neighbour_test(){
        let height_map = 
            vec![
                vec![2,2,2],
                vec![2,0,2],
                vec![2,1,2]
            ];
        let map = Matrix::new(height_map);    
        let idx = 4;    
        // let neighbour_indices = map.get_neighbour_idx(idx);
        let outer_indices = map.get_edge_indices();
        let known = DoubleMap::new(outer_indices.iter().map(|i| (*i, map.data[*i])).collect::<HashMap<usize,i32>>());
        let neighbours = map.get_candidates_from_neighbours(idx, &known);
        let min_neighbour = Matrix::get_min_neighbour(&neighbours);
        assert_eq!(min_neighbour,Some(1))
    }
    // #[test]
    fn min_neighbour_test2(){
        let height_map = 
            vec![
                vec![2,2,2],
                vec![2,0,2],
                vec![2,0,2],
                vec![2,0,2]
            ];
        let map = Matrix::new(height_map);    
        let idx = 4;    
        // let neighbour_indices = map.get_neighbour_idx(idx);
        let outer_indices = map.get_edge_indices();
        let mut known = DoubleMap::new(outer_indices.iter().map(|i| (*i, map.data[*i])).collect::<HashMap<usize,i32>>());
        known.insert(7,0);
        known.insert(10,0);
        let mut neighbours = map.get_candidates_from_neighbours(idx, &known);
        let min_neighbour = Matrix::get_min_neighbour(&neighbours);
        assert_eq!(min_neighbour,Some(0))
    }
    #[test]
    fn min_neighbour_test3(){
        let height_map = 
            vec![
                vec![2,2,2],
                vec![2,0,2],
                vec![2,0,2],
                vec![2,1,2]
            ];
        let map = Matrix::new(height_map);    
        let idx = 4;    
        let outer_indices = map.get_edge_indices();
        let mut known = DoubleMap::new(outer_indices.iter().map(|i| (*i, map.data[*i])).collect::<HashMap<usize,i32>>());
        known.insert(7,1);
        known.insert(10,1);
        let neighbours = map.get_candidates_from_neighbours(idx, &known);
        println!("{neighbours:?}");
        let min_neighbour = Matrix::get_min_neighbour(&neighbours);
        assert_eq!(min_neighbour,Some(1))
    }
    #[test]
    fn get_edges_test(){
        let height_map = vec![vec![1,1],vec![1,1]];
        let map = Matrix::new(height_map);
        let mut edges = map.get_edge_indices();
        edges.sort();
        assert_eq!(edges, vec![0,1,2,3]);
    }
    #[test]
    fn get_edges_test2(){
        let height_map = vec![vec![1,1,1,1],vec![1,0,0,1],vec![1,0,0,1],vec![1,1,1,1]];
        let map = Matrix::new(height_map);
        let mut edges = map.get_edge_indices();
        edges.sort();
        assert_eq!(edges, vec![0,1,2,3,4,7,8,11,12,13,14,15]);
    }
    // #[test]
    fn partial_sneaky_test(){
        let height_map = vec![vec![12,13,1,12],vec![13,4,13,12],vec![13,8,10,12],vec![12,13,12,12],vec![13,13,13,13]];
        let map = Matrix::new(height_map);
        let surrounding_wall_height = map.clone().find_smallest_wall();
        let volume_trapped_water = surrounding_wall_height.subtract_walls(&map).sum();
    }
    #[test]
    fn sneaky_test(){
        let height_map = vec![
            vec![12,13,1,12],
            vec![13,4,13,12],
            vec![13,8,10,12],
            vec![12,13,12,12],
            vec![13,13,13,13]];
        assert_eq!(trap_rain_water(height_map),14)
    }

    #[test]
    fn trivial(){
        let height_map = vec![
            vec![1,4,3,1,3,2],
            vec![2,3,3,2,3,1]];
        assert_eq!(trap_rain_water(height_map),0)
    }
    #[test]
    fn minimal1(){
        let height_map = vec![
            vec![2,2,2],
            vec![2,0,2],
            vec![2,2,2]];
        assert_eq!(trap_rain_water(height_map),2)
    }
    #[test]
    fn minimal2(){
        let height_map = vec![
            vec![2,2,2],
            vec![2,0,2],
            vec![2,0,2],
            vec![2,2,2]];
        assert_eq!(trap_rain_water(height_map),4)
    }
    fn minimal3(){
        let height_map = vec![
            vec![2,2,2],
            vec![2,0,2],
            vec![2,0,2],
            vec![2,1,2]];
        assert_eq!(trap_rain_water(height_map),2)
    }
    #[test]
    fn basic1(){
        let height_map = vec![
            vec![1,4,3,1,3,2],
            vec![3,2,1,3,2,4],
            vec![2,3,3,2,3,1]];
        assert_eq!(trap_rain_water(height_map),4)
    }
    #[test]
    fn basic2(){
        let height_map = vec![
            vec![3,3,3,3,3],
            vec![3,2,2,2,3],
            vec![3,2,1,2,3],
            vec![3,2,2,2,3],
            vec![3,3,3,3,3]];
        assert_eq!(trap_rain_water(height_map),10)
    }
}