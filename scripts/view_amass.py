from im.utils import *


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--npz",
        type=str,
        required=False,
        default="res/motions/single1.npz",
        help="Path to the npz file containing the AMASS data.",
    )
    arg = parser.parse_args()
    filename = osp.basename(arg.npz).split(".")[0]

    seq_amass = SMPLSequence.from_amass(
        npz_data_path=arg.npz,
        fps_out=60.0,
        color=MODEL_COLOR,
        name=f"AMASS {filename}",
        show_joint_angles=True,
    )

    v = Viewer()
    v.run_animations = True
    v.scene.camera.position = np.array([10.0, 2.5, 0.0])
    v.scene.add(seq_amass)
    v.run()
