import trimesh
from pathlib import Path
from lxml import etree
import os
import shutil
import argparse
from trimesh.decomposition import convex_decomposition


def install_vhacd():
    response = input("ONLY PROCEED IF THE SIYSTEM IS LINUX\nProceed? (N/y):")
    if response != "y":
        exit()
    cmd = """
    set -xe;
    rm -f testVHACD;
    wget https://github.com/mikedh/v-hacd-1/raw/master/bin/linux/testVHACD;
    echo "e1e79b2c1b274a39950ffc48807ecb0c81a2192e7d0993c686da90bd33985130  testVHACD" | sha256sum --check;
    chmod +x testVHACD;
    sudo mv testVHACD /usr/bin/;
    """
    os.system(cmd)


def cmd_process_meshes(args=None):
    VHACD_EXECUTABLE = shutil.which("testVHACD")

    parser = argparse.ArgumentParser(description="Convert STL to MJCF")
    parser.add_argument(
        "--folder", type=str, required=True, help="Folder with STL files"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=100_000,
        help="Maximum number of voxels generated during the voxelization stage(range=10, 000-16, 000, 000)",
    )
    parser.add_argument(
        "--maxhulls",
        type=int,
        default=64,
        help="Maximum number of convex hulls to produce",
    )
    parser.add_argument(
        "--concavity",
        type=float,
        default=0.0025,
        help="Maximum allowed concavity(range=0.0-1.0)",
    )
    parser.add_argument(
        "--planeDownsampling",
        type=int,
        default=4,
        help='Controls the granularity of the search for the "best" clipping plane(range=1-16)',
    )
    parser.add_argument(
        "--convexhullDownsampling",
        type=int,
        default=4,
        help="Controls the precision of the convex-hull generation process during the clipping plane selection stage(range=1-16)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Controls the bias toward clipping along symmetry planes(range=0.0-1.0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Controls the bias toward clipping along revolution axes(range=0.0-1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.00125,
        help="Controls the maximum allowed concavity during the merge stage(range=0.0-1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Controls the bias toward maximaxing local concavity(range=0.0-1.0)",
    )
    parser.add_argument(
        "--maxNumVerticesPerCH",
        type=int,
        default=64,
        help="Controls the maximum number of triangles per convex-hull(range=4-1024)",
    )
    parser.add_argument(
        "--minVolumePerCH",
        type=float,
        default=0.0001,
        help="Controls the adaptive sampling of the generated convex-hulls(range=0.0-0.01)",
    )
    args = vars(parser.parse_args())

    if VHACD_EXECUTABLE is None:
        print("VHACD executable not found. Please install VHACD.")
        install_vhacd()

    files_to_process = []
    for file in os.listdir(args["folder"]):
        if file.endswith(".stl"):
            files_to_process.append(file)

    for file in files_to_process:
        print(f"Processing {file}")
        mesh_name = file.split(".")[0]
        mesh_path = os.path.join(args["folder"], file)
        mesh = trimesh.load_mesh(mesh_path)

        output_folder_path = os.path.join(args["folder"], mesh_name)

        if os.path.exists(output_folder_path):
            print("Output directory already exists. Do you want to overwrite it?")
            answer = input("[y/n] ")
            if answer == "y":
                shutil.rmtree(output_folder_path, ignore_errors=True)
            else:
                exit()

        os.makedirs(output_folder_path)

        print(f"Decomposing {mesh_name}...")
        collision_hulls = convex_decomposition(
            mesh=mesh,
            debug=True,
            resolution=args["resolution"],
            maxhulls=args["maxhulls"],
            concavity=args["concavity"],
            planeDownsampling=args["planeDownsampling"],
            convexhullDownsampling=args["convexhullDownsampling"],
            alpha=args["alpha"],
            beta=args["beta"],
            gamma=args["gamma"],
            delta=args["delta"],
            maxNumVerticesPerCH=args["maxNumVerticesPerCH"],
            minVolumePerCH=args["minVolumePerCH"],
        )

        print(f"Decomposition complete. {len(collision_hulls)} hulls created.")

        print("Creating MJCF...")

        root = etree.Element("mujoco")
        root.set("model", mesh_name)
        compiler = etree.SubElement(root, "compiler")
        compiler.set("meshdir", "meshes")

        default_top = etree.SubElement(root, "default")
        default_class = etree.SubElement(default_top, "default")
        default_class.set("class", mesh_name)
        default_geom = etree.SubElement(default_class, "geom")
        default_geom.set("type", "mesh")
        default_geom.set("rgba", "0.8 0.8 0.8 1")
        default_geom.set("contype", "1")

        asset = etree.SubElement(root, "asset")
        worldbody = etree.SubElement(root, "worldbody")
        body = etree.SubElement(worldbody, "body")
        body.set("name", mesh_name)

        mesh_asset = etree.Element("mesh")
        mesh_asset.set("name", f"{mesh_name}")
        mesh_asset.set("file", f"{mesh_name}/{mesh_name}.stl")
        asset.append(mesh_asset)
        mesh.export(f"{output_folder_path}/visual.stl")

        geom = etree.Element("geom")
        geom.set("type", "mesh")
        geom.set("mesh", f"{mesh_name}")
        geom.set("rgba", "0.8 0.8 0.8 1")
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        body.append(geom)

        for i in range(len(collision_hulls)):
            convex_hull = collision_hulls[i]
            convex_hull_name = f"{mesh_name}_hull_{i}"
            convex_hull_mesh = etree.Element("mesh")
            convex_hull_mesh.set("name", convex_hull_name)
            convex_hull_mesh.set("file", f"{mesh_name}/{convex_hull_name}.stl")
            asset.append(convex_hull_mesh)
            convex_hull.export(f"{output_folder_path}/{convex_hull_name}.stl")

            convex_hull_geom = etree.Element("geom")
            convex_hull_geom.set("type", "mesh")
            convex_hull_geom.set("mesh", convex_hull_name)
            convex_hull_geom.set("class", mesh_name)
            body.append(convex_hull_geom)

        tree = etree.ElementTree(root)
        etree.indent(tree, space="  ", level=0)
        xml_path = Path(args["folder"]) / f"{mesh_name}.xml"
        with open(xml_path, "wb") as files:
            tree.write(files)


if __name__ == "__main__":
    cmd_process_meshes()
