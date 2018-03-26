import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import pdb

def testRot(p1):
   
    tmp = np.zeros(3)
    tmp[0] = np.linalg.norm(p1)
    tmp[1] = np.arccos(p1[2] / tmp[0])
    tmp[2] = np.arctan(p1[1]/p1[0])
    p1 = tmp

    # mesh sphere
    theta, phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)
    THETA, PHI = np.meshgrid(theta, phi)
    R = p1[0]
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)

    # points surrounding original pos.
    phi = np.cos(np.linspace(-np.pi, np.pi, 8)) * 0.2 + p1[2] 
    theta = np.sin(np.linspace(-np.pi, np.pi, 8)) * 0.2 + p1[1] 
    r = p1[0] / np.logspace(0, 0.8, 6)
    x = np.zeros((len(r), len(phi)))
    y = np.zeros((len(r), len(phi)))
    z = np.zeros((len(r), len(phi)))
    for j in range(len(r)):
        x[j] = r[j] * np.sin(theta) * np.cos(phi)
        y[j] = r[j] * np.sin(theta) * np.sin(phi)
        z[j] = r[j] * np.cos(theta)
    
    # halo point
    r_h, theta_h, phi_h = p1[0], p1[1], p1[2]
    x_h = r_h * np.sin(theta_h) * np.cos(phi_h)
    y_h = r_h * np.sin(theta_h) * np.sin(phi_h)
    z_h = r_h * np.cos(theta_h)
    v1 = [x_h, y_h, z_h]

    # new halo point
    r_h2, theta_h2, phi_h2 = p1[0], np.pi/2, 0
    x_h2 = r_h2 * np.sin(theta_h2) * np.cos(phi_h2)
    y_h2 = r_h2 * np.sin(theta_h2) * np.sin(phi_h2)
    z_h2 = r_h2 * np.cos(theta_h2)
    vrot = [x_h2, y_h2, z_h2]

    # angle between new and old points
    angle = np.arccos(np.dot(v1, vrot) / np.dot(np.linalg.norm(v1), np.linalg.norm(vrot)))
    t = angle
    a = [x_h, y_h, z_h]
    b = [x_h2, y_h2, z_h2]
    k = np.cross(a, b) / (np.prod(np.linalg.norm([a, b], axis=1)) * np.sin(t))

    # rotate all other points
    x2 = np.zeros(np.shape(x))
    y2 = np.zeros(np.shape(x))
    z2 = np.zeros(np.shape(x))
    for j in range(len(r)):
        for i in range(len(phi)):
            vp = np.array([x[j][i], y[j][i], z[j][i]])
            vprot = vp * np.cos(t) + np.cross(k, vp)*np.sin(t) + k*(np.dot(k, vp))*(1-np.cos(t))

            x2[j][i] = vprot[0]
            y2[j][i] = vprot[1]
            z2[j][i] = vprot[2]

    print("begin: {}".format(a))
    print("end: {}".format(b))
    print("axb: {}".format(np.cross(a, b)))
    print("mag axb: {}".format(np.linalg.norm(np.cross(a,b))))
    print("k: {}".format(np.cross(a, b) / np.linalg.norm(np.cross(a,b))))
    print("k x v: {}".format(np.cross(k, a)))
    print("k . v: {}".format(np.dot(k, a)))
    print("B: {}".format(angle))
    print("result: {}".format(vprot))

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, lw=0.6, color='k', alpha=0.4)
    for n in range(len(r)):
        ax.plot(x[n], y[n], z[n], '.b')
        ax.plot(x2[n], y2[n], z2[n], '.g')
    ax.scatter(x_h, y_h, z_h, color='red')
    ax.scatter(x_h2, y_h2, z_h2, color='m')
    ax.set_xlabel('x')

    plt.show()
