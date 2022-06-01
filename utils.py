import math
import numpy as np


def write_ply_point(name, vertices):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    fout.close()


def write_ply_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face "+str(len(triangles))+"\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
    fout.close()


def write_ply_point_normal(name, vertices, normals=None):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("end_header\n")
    if normals is None:
        for ii in range(len(vertices)):
            fout.write(
                str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + " " + str(vertices[ii, 3]) + " " + str(
                    vertices[ii, 4]) + " " + str(vertices[ii, 5]) + "\n")
    else:
        for ii in range(len(vertices)):
            fout.write(
                str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + " " + str(normals[ii, 0]) + " " + str(
                    normals[ii, 1]) + " " + str(normals[ii, 2]) + "\n")
    fout.close()


def sample_points_triangle(vertices, triangles, num_of_points):
    epsilon = 1e-6
    triangle_area_list = np.zeros([len(triangles)], np.float32)
    triangle_normal_list = np.zeros([len(triangles), 3], np.float32)
    for i in range(len(triangles)):
        # area = |u x v|/2 = |u||v|sin(uv)/2
        a, b, c = vertices[triangles[i, 1]] - vertices[triangles[i, 0]]
        x, y, z = vertices[triangles[i, 2]] - vertices[triangles[i, 0]]
        ti = b * z - c * y
        tj = c * x - a * z
        tk = a * y - b * x
        area2 = math.sqrt(ti * ti + tj * tj + tk * tk)
        if area2 < epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i, 0] = 0
            triangle_normal_list[i, 1] = 0
            triangle_normal_list[i, 2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i, 0] = ti / area2
            triangle_normal_list[i, 1] = tj / area2
            triangle_normal_list[i, 2] = tk / area2

    triangle_area_sum = np.sum(triangle_area_list)
    sample_prob_list = (num_of_points / triangle_area_sum) * triangle_area_list

    triangle_index_list = np.arange(len(triangles))

    point_normal_list = np.zeros([num_of_points, 6], np.float32)
    count = 0
    watchdog = 0

    while count < num_of_points:
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog > 100:
            print("infinite loop here!")
            return point_normal_list
        for i in range(len(triangle_index_list)):
            if count >= num_of_points:
                break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = int(prob)
            prob_f = prob - prob_i
            if np.random.random() < prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb, 1]] - vertices[triangles[dxb, 0]]
            v = vertices[triangles[dxb, 2]] - vertices[triangles[dxb, 0]]
            base = vertices[triangles[dxb, 0]]
            for j in range(prob_i):
                # sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x + v_y >= 1:
                    u_x = 1 - u_x
                    v_y = 1 - v_y
                ppp = u * u_x + v * v_y + base

                point_normal_list[count, :3] = ppp
                point_normal_list[count, 3:] = normal_direction
                count += 1
                if count >= num_of_points: break

    return point_normal_list


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def plot_3d_point_to_image(vertices):
    if vertices.shape[1] == 6:
        vertices = vertices[:, :3]
    img = np.ones((256, 256), np.uint8) * 255
    # rotation around z axis, y axis, x axis
    view_projection_matrix = np.matmul(np.matmul(rotation_matrix([0,0,1], -180*3.14/180), rotation_matrix([0,1,0], -30*3.14/180)),
                                       rotation_matrix([1,0,0], -30*3.14/180))
    xyz = np.matmul(vertices, view_projection_matrix)
    x = xyz[:, 0]
    y = xyz[:, 1]
    x = ((x + 0.5) * 256).astype(int)
    y = ((y + 0.5) * 256).astype(int)
    x[x > 255] = 255
    y[y > 255] = 255
    x[x < 0] = 0
    y[y < 0] = 0

    for i in range(x.shape[0]):
        img[y[i], x[i]] = 0

    return img
