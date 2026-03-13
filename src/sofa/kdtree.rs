//! KD-tree for efficient 3D nearest neighbor search.
//!
//! This is a simple 3-dimensional KD-tree implementation optimized for
//! finding the nearest HRTF filter position.

/// A 3D point.
pub type Point3 = [f32; 3];

/// A node in the KD-tree.
#[derive(Debug)]
struct KdNode {
    /// Position of this node.
    pos: Point3,
    /// The dimension used for splitting at this node (0=x, 1=y, 2=z).
    split_dim: usize,
    /// Data associated with this node, typically a filter index.
    data: usize,
    /// Left subtree, containing values less than the split.
    left: Option<Box<KdNode>>,
    /// Right subtree, containing values greater than or equal to the split.
    right: Option<Box<KdNode>>,
}

/// Bounding box for the KD-tree.
#[derive(Debug, Clone)]
struct BoundingBox {
    min: Point3,
    max: Point3,
}

impl BoundingBox {
    fn new(pos: &Point3) -> Self {
        Self {
            min: *pos,
            max: *pos,
        }
    }

    fn extend(&mut self, pos: &Point3) {
        for (i, &p) in pos.iter().enumerate().take(3) {
            if p < self.min[i] {
                self.min[i] = p;
            }
            if p > self.max[i] {
                self.max[i] = p;
            }
        }
    }

    /// Compute squared distance from point to this bounding box.
    fn dist_sq(&self, pos: &Point3) -> f32 {
        let mut result = 0.0;
        for (i, &p) in pos.iter().enumerate().take(3) {
            if p < self.min[i] {
                result += (self.min[i] - p).powi(2);
            } else if p > self.max[i] {
                result += (self.max[i] - p).powi(2);
            }
        }
        result
    }
}

/// A 3-dimensional KD-tree for efficient nearest neighbor search.
#[derive(Debug)]
pub struct KdTree {
    root: Option<Box<KdNode>>,
    bounds: Option<BoundingBox>,
}

impl Default for KdTree {
    fn default() -> Self {
        Self::new()
    }
}

impl KdTree {
    /// Create a new empty KD-tree.
    pub fn new() -> Self {
        Self {
            root: None,
            bounds: None,
        }
    }

    /// Insert a point with associated data into the tree.
    ///
    /// # Arguments
    /// * `pos` - The 3D position [x, y, z]
    /// * `data` - The data to associate, typically a filter index
    pub fn insert(&mut self, pos: Point3, data: usize) {
        // Update bounding box
        match &mut self.bounds {
            Some(bounds) => bounds.extend(&pos),
            None => self.bounds = Some(BoundingBox::new(&pos)),
        }

        // Insert into tree
        Self::insert_rec(&mut self.root, pos, data, 0);
    }

    fn insert_rec(node: &mut Option<Box<KdNode>>, pos: Point3, data: usize, depth: usize) {
        let split_dim = depth % 3;

        match node {
            None => {
                *node = Some(Box::new(KdNode {
                    pos,
                    split_dim,
                    data,
                    left: None,
                    right: None,
                }));
            }
            Some(n) => {
                if pos[n.split_dim] < n.pos[n.split_dim] {
                    Self::insert_rec(&mut n.left, pos, data, depth + 1);
                } else {
                    Self::insert_rec(&mut n.right, pos, data, depth + 1);
                }
            }
        }
    }

    /// Find the nearest neighbor to the given position.
    ///
    /// Returns the data associated with the nearest point, or None if tree is empty.
    pub fn nearest(&self, pos: &Point3) -> Option<usize> {
        let root = self.root.as_ref()?;
        let bounds = self.bounds.as_ref()?;

        let mut best_node = root.as_ref();
        let mut best_dist_sq = Self::dist_sq(&root.pos, pos);

        // Working copy of bounds for search
        let mut rect = bounds.clone();

        Self::nearest_rec(root, pos, &mut best_node, &mut best_dist_sq, &mut rect);

        Some(best_node.data)
    }

    fn nearest_rec<'a>(
        node: &'a KdNode,
        pos: &Point3,
        best: &mut &'a KdNode,
        best_dist_sq: &mut f32,
        rect: &mut BoundingBox,
    ) {
        let dim = node.split_dim;
        let diff = pos[dim] - node.pos[dim];

        // Determine which subtree is nearer
        let go_left = diff <= 0.0;

        // Recurse into nearer subtree
        let nearer = if go_left { &node.left } else { &node.right };
        if let Some(nearer_node) = nearer {
            let old_coord = if go_left {
                rect.max[dim]
            } else {
                rect.min[dim]
            };
            if go_left {
                rect.max[dim] = node.pos[dim];
            } else {
                rect.min[dim] = node.pos[dim];
            }
            Self::nearest_rec(nearer_node, pos, best, best_dist_sq, rect);
            if go_left {
                rect.max[dim] = old_coord;
            } else {
                rect.min[dim] = old_coord;
            }
        }

        // Check current node
        let dist_sq = Self::dist_sq(&node.pos, pos);
        if dist_sq < *best_dist_sq {
            *best = node;
            *best_dist_sq = dist_sq;
        }

        // Check if we need to search farther subtree
        let farther = if go_left { &node.right } else { &node.left };
        if let Some(farther_node) = farther {
            let old_coord = if go_left {
                rect.min[dim]
            } else {
                rect.max[dim]
            };
            if go_left {
                rect.min[dim] = node.pos[dim];
            } else {
                rect.max[dim] = node.pos[dim];
            }

            // Only search if the hyperrect could contain a closer point
            if rect.dist_sq(pos) < *best_dist_sq {
                Self::nearest_rec(farther_node, pos, best, best_dist_sq, rect);
            }

            if go_left {
                rect.min[dim] = old_coord;
            } else {
                rect.max[dim] = old_coord;
            }
        }
    }

    /// Compute squared distance between two points.
    #[inline]
    fn dist_sq(a: &Point3, b: &Point3) -> f32 {
        (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
    }

    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let tree = KdTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.nearest(&[0.0, 0.0, 0.0]), None);
    }

    #[test]
    fn test_single_point() {
        let mut tree = KdTree::new();
        tree.insert([1.0, 2.0, 3.0], 42);

        assert!(!tree.is_empty());
        assert_eq!(tree.nearest(&[0.0, 0.0, 0.0]), Some(42));
        assert_eq!(tree.nearest(&[1.0, 2.0, 3.0]), Some(42));
        assert_eq!(tree.nearest(&[100.0, 100.0, 100.0]), Some(42));
    }

    #[test]
    fn test_multiple_points() {
        let mut tree = KdTree::new();
        tree.insert([0.0, 0.0, 0.0], 0);
        tree.insert([1.0, 0.0, 0.0], 1);
        tree.insert([0.0, 1.0, 0.0], 2);
        tree.insert([0.0, 0.0, 1.0], 3);

        // Origin should find index 0
        assert_eq!(tree.nearest(&[0.0, 0.0, 0.0]), Some(0));

        // Point closest to [1, 0, 0] should find index 1
        assert_eq!(tree.nearest(&[0.9, 0.0, 0.0]), Some(1));

        // Point closest to [0, 1, 0] should find index 2
        assert_eq!(tree.nearest(&[0.0, 0.9, 0.0]), Some(2));

        // Point closest to [0, 0, 1] should find index 3
        assert_eq!(tree.nearest(&[0.0, 0.0, 0.9]), Some(3));
    }

    #[test]
    fn test_find_nearest_among_many() {
        let mut tree = KdTree::new();

        // Create a grid of points
        let mut idx = 0;
        for x in -5..=5 {
            for y in -5..=5 {
                for z in -5..=5 {
                    tree.insert([x as f32, y as f32, z as f32], idx);
                    idx += 1;
                }
            }
        }

        // Test that origin finds [0, 0, 0]
        let center_idx = 5 * 11 * 11 + 5 * 11 + 5; // Index of [0, 0, 0]
        assert_eq!(tree.nearest(&[0.1, 0.1, 0.1]), Some(center_idx));

        // Test corner
        let corner_idx = 0; // Index of (-5, -5, -5)
        assert_eq!(tree.nearest(&[-4.9, -4.9, -4.9]), Some(corner_idx));
    }

    #[test]
    fn test_bounding_box() {
        let mut bb = BoundingBox::new(&[0.0, 0.0, 0.0]);
        assert_eq!(bb.min, [0.0, 0.0, 0.0]);
        assert_eq!(bb.max, [0.0, 0.0, 0.0]);

        bb.extend(&[1.0, -1.0, 2.0]);
        assert_eq!(bb.min, [0.0, -1.0, 0.0]);
        assert_eq!(bb.max, [1.0, 0.0, 2.0]);

        // Point inside box has distance 0
        assert_eq!(bb.dist_sq(&[0.5, -0.5, 1.0]), 0.0);

        // Point outside box
        assert!((bb.dist_sq(&[2.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
    }
}
