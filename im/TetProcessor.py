import numpy as np

from im.utils import *


class TetProcessor:
    def __init__(self, p1_jpos: np.ndarray, p2_jpos: np.ndarray, **kwargs):
        """
        :param p1_jpos: (f, j_idx, 3) np.ndarray, the joint positions of the first person.
        :param p2_jpos: (f, j_idx, 3) np.ndarray, the joint positions of the second person.
        :param kwargs:
        """
        if p1_jpos.ndim == 2:
            p1_jpos = p1_jpos[np.newaxis, ...]
            p2_jpos = p2_jpos[np.newaxis, ...]

        assert p1_jpos.shape[0] == p2_jpos.shape[0], "The number of frame should be the same."
        zeros = np.zeros((p1_jpos.shape[0], 1, 3))
        self._jpos = np.concatenate((p1_jpos, p2_jpos, zeros), axis=1)
        self._jnum = self._jpos.shape[1] - 1

        self._whitelist = None
        if kwargs.get("use_whitelist_filter", True):
            self._init_whitelist()

        self._use_interaction_filter = kwargs.get("use_interaction_filter", True)

    def compute(self) -> np.ndarray:
        simplices = self._compute_delaunay()
        tet_indices = self._concat_tet_indices(simplices)
        adj_list = self._build_adjacency_list(tet_indices, self._jnum)
        self._compute_laplacian(adj_list)

        return self._convert_tet_edge_coord(tet_indices)

    def _compute_delaunay(self) -> List[np.ndarray]:
        """
        Compute the Delaunay tetrahedralization.
        :return: List of (n, 4) np.ndarray, the indices of the vertices of the tetrahedral.
        """
        num_frames = self._jpos.shape[0]

        start = time()
        with mp.Pool(mp.cpu_count()) as pool:
            simplices = pool.map(self._run_delaunay, [self._jpos[frame, :-1] for frame in range(num_frames)])
        print(f"[Compute] {num_frames} tet took {time() - start:.2f} seconds.")

        return simplices

    def _compute_laplacian(self, adj_list: List[List[Set[int]]]) -> np.ndarray:
        """
        :param adj_list: refer to _build_adjacency_list
        :return: (f, n, 3) np.ndarray, the laplacian coordinates of the vertices.
            If the vertices have no neighbor, remain the coordinate to be -1.
        """
        laplacian_coord = np.full_like(self._jpos, -1)
        for f, frame_adj_list in enumerate(adj_list):
            for i in range(self._jnum):
                neighbor_idx = list(frame_adj_list[i])
                if len(neighbor_idx) == 0:
                    continue
                centroid = np.mean(self._jpos[f, neighbor_idx], axis=0)
                laplacian_coord[f, i] = self._jpos[f, i] - centroid
        return laplacian_coord

    def _convert_tet_edge_coord(self, tet_indices: np.ndarray) -> np.ndarray:
        """

        :param tet_indices: (f, n, 4) np.ndarray, the indices of the vertices of the tetrahedral.
        :return: (f, n * 12, 3) np.ndarray, the coordinates of the edges of the tetrahedral.
        """
        all_start_coords = []
        all_end_coords = []

        for frame in range(self._jpos.shape[0]):
            # Get the tetrahedra indices for the current frame
            current_tetrahedra = tet_indices[frame]

            # # Filter out invalid tetrahedra (those with any index as -1)
            # valid_mask = np.all(current_tetrahedra != -1, axis=1)
            # valid_tetrahedra = current_tetrahedra[valid_mask]

            # Extract edges
            start_indices, end_indices = self._extract_edges(current_tetrahedra)

            # Map indices to coordinates
            start_coords = self._jpos[frame, start_indices]
            end_coords = self._jpos[frame, end_indices]

            all_start_coords.append(start_coords)
            all_end_coords.append(end_coords)

        all_start_coords = np.stack(all_start_coords, axis=0)
        all_end_coords = np.stack(all_end_coords, axis=0)

        line_shape = list(all_start_coords.shape)
        line_shape[1] *= 2
        line_pos = np.full(line_shape, -1, dtype=all_start_coords.dtype)
        line_pos[:, ::2, :] = all_start_coords
        line_pos[:, 1::2, :] = all_end_coords
        return line_pos

    def _init_whitelist(self):
        """
        Initialize the whitelist of vertices to keep.
        If the tetrahedral contains any of the vertices in the whitelist, keep it.
        :return:
        """
        whitelist = [
            # 0
            0, 22,  # root
            15, 37,  # head
            20, 21, 42, 43,  # wrist
            10, 11, 32, 33,  # foot
        ]
        self._whitelist = set(whitelist)

    def _run_delaunay(self, jpos: np.ndarray) -> np.ndarray:
        """
        Run the Delaunay tetrahedralization.
        :param jpos: (j_idx, 3) np.ndarray, the joint positions.
        :return: (n, 4) np.ndarray, the indices of the vertices of the tetrahedral.
        """
        delaunay = Delaunay(jpos)
        tetrahedral = delaunay.simplices

        if self._use_interaction_filter:
            filtered_tet = []
            for tet in tetrahedral:
                vertex_set = set(tet)
                if not (all_smaller_than(vertex_set, 22) or all_larger_than(vertex_set, 21)):
                    filtered_tet.append(tet)
            tetrahedral = filtered_tet

        if self._whitelist:
            filtered_tet = []
            for tet in tetrahedral:
                vertex_set = set(tet)
                if vertex_set & self._whitelist:
                    filtered_tet.append(tet)
            tetrahedral = filtered_tet

        return np.array(tetrahedral)

    @staticmethod
    def _build_adjacency_list(tet_indices: np.ndarray, num_vertices: int) -> List[List[Set[int]]]:
        """
        :param tet_indices: (f, n, 4) np.ndarray, the indices of the vertices of the tetrahedral.
        :param num_vertices: int, the number of vertices.
        :return: List of List of Set, the adjacency list of the vertices.
            First index is the frame index.
            Second index is the vertex index.
            Element is the set of the index of adjacent vertices.
        """
        adj_list = []
        for f in range(tet_indices.shape[0]):
            frame_adj_list = [set() for _ in range(num_vertices)]
            for tet in tet_indices[f]:
                if tet[0] == -1:
                    continue
                for i in range(4):
                    for j in range(i + 1, 4):
                        frame_adj_list[tet[i]].add(tet[j])
                        frame_adj_list[tet[j]].add(tet[i])
            adj_list.append(frame_adj_list)
        return adj_list

    @staticmethod
    def _concat_tet_indices(simplices: List[np.ndarray]) -> np.ndarray:
        """
        1. Pad the tetrahedral indices to make them have the same length.
        2. Concatenate the tetrahedral indices.
        :param simplices: list, the list of tetrahedral indices.
        :return: (f, n, 4) np.ndarray, the concatenated tetrahedral indices.
        """
        max_len = max([len(tet) for tet in simplices])
        for i, tet in enumerate(simplices):
            pad = np.full((max_len - len(tet), 4), -1)
            if len(tet) == 0:
                simplices[i] = pad
            elif len(tet) < max_len:
                simplices[i] = np.concatenate((tet, pad), axis=0)

        return np.stack(simplices, axis=0)

    @staticmethod
    def _extract_edges(tetrahedra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract edges from a given tetrahedra array.

        :param tetrahedra: A (n, 4) array of tetrahedra indices.
        :return: A (n*6, 2) array of edges.
        """
        edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        start_indices = tetrahedra[:, edges[:, 0]]
        end_indices = tetrahedra[:, edges[:, 1]]

        return start_indices.flatten(), end_indices.flatten()
