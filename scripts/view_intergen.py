from im.utils import *


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--pkl",
        type=str,
        required=False,
        default="res/motions/inter1.pkl",
        help="Path to the pkl file containing the INTERGEN data.",
    )
    arg = parser.parse_args()
    filename = osp.basename(arg.pkl).split(".")[0]

    seq1, seq2 = SMPLSequence.from_intergen(
        pkl_data_path=arg.pkl,
        fps_out=60.0,
        color=MESH_COLOR,
        name=f"InterGen {filename}",
        show_joint_angles=True,
    )

    line_shape = list(seq1.joints.shape)
    line_shape[1] *= 2
    line_pos = np.zeros(line_shape)
    line_pos[:, ::2, :] = seq1.joints
    line_pos[:, 1::2, :] = seq2.joints
    line_pos_z_up = line_pos[:, :, [0, 2, 1]]
    line_renderable = Lines(line_pos, color=LINE_COLOR, r_base=0.004)

    # Display in the viewer.
    v = Viewer()
    v.run_animations = True
    v.scene.camera.position = np.array([1.5, 2, 2.5])
    v.scene.add(seq1, seq2)
    v.scene.add(line_renderable)
    v.run()
