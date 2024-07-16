import pyvista as pv
from im.utils import *

"""
    Conduct the tetrahedralization
"""

whitelist = [
    0, 22,  # root
    15, 37,  # head
    20, 21, 42, 43,  # wrist
    10, 11, 32, 33,  # foot
]

# Perform Delaunay tetrahedralization
vertices = np.load('res/inter_jpos.npy')
start = time()
delaunay = Delaunay(vertices)
tetrahedra = delaunay.simplices

"""
    Filter the mesh
"""

whitelist = set(whitelist)
filtered_tetrahedra = []
for tet in tetrahedra:
    vertex_set = set(tet)
    if not (all_smaller_than(vertex_set, 22) or all_larger_than(vertex_set, 21)):
        if vertex_set & whitelist:
            filtered_tetrahedra.append(tet)

print(f"There are {len(tetrahedra)} tetrahedra in the mesh.")
print(f"There are {len(filtered_tetrahedra)} tetrahedra in the mesh after filtering.")
print(f"Computation took {time() - start:.6f} seconds.")

tetrahedra = np.array(filtered_tetrahedra)

"""
    Visualize the mesh
"""

# Create a PyVista mesh
mesh = pv.PolyData()

# Add vertices
mesh.points = vertices

# Correctly add tetrahedra faces
faces = []
for tet in filtered_tetrahedra:
    faces.append([3, tet[0], tet[1], tet[2]])
    faces.append([3, tet[0], tet[1], tet[3]])
    faces.append([3, tet[0], tet[2], tet[3]])
    faces.append([3, tet[1], tet[2], tet[3]])
mesh.faces = np.hstack(faces)

# Create a plotter object
plotter = pv.Plotter()

# Add the mesh to the plotter with translucent faces
plotter.add_mesh(mesh, color='cyan', opacity=0.3, show_edges=True, edge_color='black', line_width=3)

sphere = pv.Sphere(radius=0.02)  # Create a sphere glyph
point_cloud = pv.PolyData(vertices)
glyphs = point_cloud.glyph(scale=False, geom=sphere)
plotter.add_mesh(glyphs, color='red')

labels = [str(i) for i in range(len(vertices))]
plotter.add_point_labels(vertices, labels, point_size=20, font_size=15, text_color="blue")
plotter.add_key_event("q", plotter.close)

# Show the plot
plotter.show()
