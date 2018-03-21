import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import pdb

def testRot(p1):
    
    # mesh sphere
    theta, phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)
    THETA, PHI = np.meshgrid(theta, phi)
    R = 1
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)

    # points surrounding original pos.
    phi = np.cos(np.linspace(-np.pi, np.pi, 8)) * 0.2 + p1[2] 
    theta = np.sin(np.linspace(-np.pi, np.pi, 8)) * 0.2 + p1[1] 
    r = p1[0]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # halo point
    r, theta_h, phi_h = p1[0], p1[1], p1[2]
    x_h = r * np.sin(theta_h) * np.cos(phi_h)
    y_h = r * np.sin(theta_h) * np.sin(phi_h)
    z_h = r * np.cos(theta_h)
    v1 = [x_h, y_h, z_h]

    # new halo point
    r, theta_h2, phi_h2 = p1[0], np.pi/2, 0
    x_h2 = r * np.sin(theta_h2) * np.cos(phi_h2)
    y_h2 = r * np.sin(theta_h2) * np.sin(phi_h2)
    z_h2 = r * np.cos(theta_h2)
    vrot = [x_h2, y_h2, z_h2]

    # angle between new and old points
    angle = np.arccos(np.dot(v1, vrot) / np.dot(np.linalg.norm(v1), np.linalg.norm(vrot)))

    # rotate all other points
    x2 = np.zeros(len(x))
    y2 = np.zeros(len(x))
    z2 = np.zeros(len(x))
    for i in range(len(phi)):
        vp = np.array([x[i], y[i], z[i]])
        #vp = np.array([x_h, y_h, z_h])
        t = angle
        a = [x_h, y_h, z_h]
        b = [x_h2, y_h2, z_h2]
        k = np.cross(a, b) / (np.prod(np.linalg.norm([a, b], axis=1)) * np.sin(t))
        vprot = vp * np.cos(t) + np.cross(k, vp)*np.sin(t) + k*(np.dot(k, vp))*(1-np.cos(t))

        x2[i] = vprot[0]
        y2[i] = vprot[1]
        z2[i] = vprot[2]
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
    ax.plot(x, y, z, '.b')
    ax.scatter(x_h, y_h, z_h, color='red')
    ax.plot(x2, y2, z2, '.g')
    ax.scatter(x_h2, y_h2, z_h2, color='m')
    ax.set_xlabel('x')
    plt.show()
