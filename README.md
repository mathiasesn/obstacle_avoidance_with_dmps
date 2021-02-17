# Obstacle Avoidance with Dynamic Movements Primitives

This project explores the abillity of performing obstacle avoidance with the use of dymamic movements primitives. The project is part of the course Project in Advanced Robotics at SDU which is a 5 ETCS course. The project consist of:

- Dynamic movement primitives
- Obstacle avoidance
- Link collision avoidance
- Pose estimation of object

For more information see the [report](Obstacle_Avoidance_with_Dynamic_Movement_Primitives.pdf) or the short [presentation](ObstacleAvoidanceWithPotentialFields.pdf).

## Content

- [DMP](DMP/) - contains the code and experiments for dynamic movement primitives for position.
- [DMP_quaternion](DMP_quaternion/) - contains the code and experiments for dynamic movement primitives for rotation.
- [LinkCollisionAvoidance](LinkCollisionAvoidance/) - contains the code and experiments for link collision avoidance.
- [pose_estimation](pose_estimation/) - contains the code and experiments of different pose estimation methods.
- [camera_calibration](camera_calibration/) - contains the code for camera calibration with ChArUco.
- [Matlab_in_Python_Example](Matlab_in_Python_Example/) - shows how to run Matlab in python.
- [latex](latex/) - contains the LaTeX for the [report](Obstacle_Avoidance_with_Dynamic_Movement_Primitives.pdf).

## Litterature

**Dynamic Movement Primitives**

- Ijspeert, A. J., Nakanishi, J., Hoffmann, H., Pastor, P., & Schaal, S. (2013). Dynamical movement primitives: learning attractor models for motor behaviors. Neural computation, 25(2), 328-373.
- Ijspeert, A. J., Nakanishi, J., & Schaal, S. (2002, May). Movement imitation with nonlinear dynamical systems in humanoid robots. In Proceedings 2002 IEEE International Conference on Robotics and Automation (Cat. No. 02CH37292) (Vol. 2, pp. 1398-1403). IEEE.

**Obstacle Avoidance**

- Park, D. H., Hoffmann, H., Pastor, P., & Schaal, S. (2008, December). Movement reproduction and obstacle avoidance with dynamic movement primitives and potential fields. In Humanoids 2008-8th IEEE-RAS International Conference on Humanoid Robots (pp. 91-98). IEEE.
- Hoffmann, H., Pastor, P., Park, D. H., & Schaal, S. (2009, May). Biologically-inspired dynamical systems for movement generation: automatic real-time goal adaptation and obstacle avoidance. In 2009 IEEE International Conference on Robotics and Automation (pp. 2587-2592). IEEE.

**Intuitive explanations and some simple Python code**

- <https://studywolf.wordpress.com/2013/11/16/dynamic-movement-primitives-part-1-the-basics/>
- <https://studywolf.wordpress.com/2016/05/13/dynamic-movement-primitives-part-4-avoiding-obstacles/>

More information is given in lecture 10: Programming by Demonstration in Advanced Robotics 2.

## Credits

The main work of this project was done by [Bjarke Larsen](https://github.com/BjarkeL), [Emil Ancker](https://github.com/ancker1), [Mathias Nielsen](https://github.com/Masle16), and [Mikkel Larsen](https://github.com/mikkellars).

The supervisor for this project was [Iñigo Iturrate San Juan](https://portal.findresearcher.sdu.dk/da/persons/inju) from SDU.

Here is a list of repositories which inspired this project:

- [DenseFusion](https://github.com/j96w/DenseFusion)
- [pydmps](https://github.com/studywolf/pydmps)

## License

Licensed under the [MIT License](LICENSE).
