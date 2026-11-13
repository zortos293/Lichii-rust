//! The mix graph: an adjacency matrix of mix connections and the depth-first topological sort
//! that orders mixes when splitters route them.

use lichii_common::align_up;

/// Directed edges between mixes, `count * count` flags.
#[derive(Clone, Debug, Default)]
pub struct EdgeMatrix {
    edges: Vec<bool>,
    count: u32,
}

impl EdgeMatrix {
    /// Work buffer bytes the reference reserves for `count` nodes.
    pub const fn work_buffer_size(count: u32) -> u64 {
        align_up(count as u64 * count as u64, 0x40) / 8
    }

    pub fn new(count: u32) -> Self {
        Self {
            edges: vec![false; count as usize * count as usize],
            count,
        }
    }

    pub fn node_count(&self) -> u32 {
        self.count
    }

    fn index(&self, id: u32, destination: u32) -> Option<usize> {
        (id < self.count && destination < self.count)
            .then(|| (self.count * id + destination) as usize)
    }

    pub fn connected(&self, id: u32, destination: u32) -> bool {
        self.index(id, destination).is_some_and(|i| self.edges[i])
    }

    pub fn connect(&mut self, id: u32, destination: u32) {
        if let Some(i) = self.index(id, destination) {
            self.edges[i] = true;
        }
    }

    pub fn disconnect(&mut self, id: u32, destination: u32) {
        if let Some(i) = self.index(id, destination) {
            self.edges[i] = false;
        }
    }

    pub fn remove_edges(&mut self, id: u32) {
        for destination in 0..self.count {
            self.disconnect(id, destination);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchState {
    Unknown,
    Found,
    Complete,
}

/// Topological sort over an [`EdgeMatrix`]. Buffers are sized once so sorting allocates nothing.
#[derive(Clone, Debug, Default)]
pub struct NodeStates {
    states: Vec<SearchState>,
    results: Vec<u32>,
    result_pos: usize,
    stack: Vec<u32>,
}

impl NodeStates {
    /// Work buffer bytes the reference reserves for `count` nodes. It counts the reference's
    /// 0x30-byte host `BitArray` per node.
    pub const fn work_buffer_size(count: u32) -> u64 {
        const HOST_BIT_ARRAY_SIZE: u64 = 0x30;
        let count = count as u64;
        (align_up(count, 0x40) / 8) * 2 + count * HOST_BIT_ARRAY_SIZE + count * count * 4
    }

    pub fn new(count: u32) -> Self {
        let count = count as usize;
        Self {
            states: vec![SearchState::Unknown; count],
            results: vec![u32::MAX; count],
            result_pos: 0,
            stack: Vec::with_capacity(count * (count + 1)),
        }
    }

    pub fn node_count(&self) -> u32 {
        self.states.len() as u32
    }

    fn reset(&mut self) {
        self.states.fill(SearchState::Unknown);
        self.results.fill(u32::MAX);
        self.result_pos = 0;
        self.stack.clear();
    }

    /// Sorts the graph. Returns false and clears the results when it has a cycle.
    pub fn tsort(&mut self, edges: &EdgeMatrix) -> bool {
        self.reset();
        for node in 0..self.node_count() {
            if self.states[node as usize] == SearchState::Unknown {
                self.stack.push(node);
            }
            while let Some(&current) = self.stack.last() {
                match self.states[current as usize] {
                    SearchState::Unknown => self.states[current as usize] = SearchState::Found,
                    SearchState::Found => {
                        self.states[current as usize] = SearchState::Complete;
                        self.results[self.result_pos] = current;
                        self.result_pos += 1;
                        self.stack.pop();
                        continue;
                    }
                    SearchState::Complete => {
                        self.stack.pop();
                        continue;
                    }
                }
                for edge in 0..edges.node_count().min(self.node_count()) {
                    if !edges.connected(current, edge) {
                        continue;
                    }
                    match self.states[edge as usize] {
                        SearchState::Unknown => self.stack.push(edge),
                        SearchState::Found => {
                            tracing::error!("audio mix graph has a cycle");
                            self.reset();
                            return false;
                        }
                        SearchState::Complete => {}
                    }
                }
            }
        }
        true
    }

    /// Sorted node ids, sinks of the graph last: the result buffer read backwards from its end,
    /// one entry per completed node.
    pub fn sorted_results(&self) -> impl Iterator<Item = u32> + '_ {
        self.results.iter().rev().take(self.result_pos).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_sources_before_destinations() {
        let mut edges = EdgeMatrix::new(4);
        edges.connect(1, 0);
        edges.connect(2, 1);
        edges.connect(3, 0);
        edges.connect(2, 3);
        let mut nodes = NodeStates::new(4);
        assert!(nodes.tsort(&edges));
        let sorted: Vec<u32> = nodes.sorted_results().collect();
        assert_eq!(sorted, vec![2, 3, 1, 0]);
    }

    #[test]
    fn cycles_fail() {
        let mut edges = EdgeMatrix::new(3);
        edges.connect(0, 1);
        edges.connect(1, 2);
        edges.connect(2, 0);
        let mut nodes = NodeStates::new(3);
        assert!(!nodes.tsort(&edges));
        assert_eq!(nodes.sorted_results().count(), 0);
        edges.remove_edges(2);
        assert!(nodes.tsort(&edges));
        assert_eq!(nodes.sorted_results().collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn work_buffer_sizes() {
        assert_eq!(EdgeMatrix::work_buffer_size(5), 8);
        assert_eq!(NodeStates::work_buffer_size(5), 356);
        assert_eq!(NodeStates::work_buffer_size(9), 772);
        assert_eq!(EdgeMatrix::work_buffer_size(9), 16);
    }
}
