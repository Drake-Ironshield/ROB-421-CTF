from aim_fsm import *
import numpy as np

zonex = -150 # back right corner from starting pose set as goal location
zoney = -150
color = OrangeBarrelObj # team color

class LocateBarrel(StateNode):
    def start(self, event=None):
        super().start(event)
        closest = None
        best_dist = float('inf')
        for obj in robot.world_map.objects.values():
            if isinstance(obj, color) and obj.is_visible:
                dx = obj.pose.x - robot.pose.x 
                dy = obj.pose.y - robot.pose.y
                dist = np.sqrt(dx**2 + dy**2)# finding distance same as rest
                # skip barrels already sitting in the drop zone
                zdx = obj.pose.x - zonex
                zdy = obj.pose.y - zoney
                zdist = np.sqrt(zdx**2 + zdy**2)
                if zdist <= 150:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    closest = obj
        if closest is None:
            self.post_failure()
        else:
            self.parent.target_barrel = closest
            #print(f"barrel is at {best_dist:.1f}")
            self.post_data(best_dist)

class TurnToBarrel(StateNode):
    def start(self, event=None):
        super().start(event)
        barrel = getattr(self.parent, 'target_barrel', None)
        if barrel is None:
            self.post_failure()
            return
        dx = barrel.pose.x - robot.pose.x
        dy = barrel.pose.y - robot.pose.y
        bearing = np.degrees(np.arctan2(dy, dx))
        turn_angle = (bearing - np.degrees(robot.pose.theta) + 180) % 360 - 180
        #print(f"turning {turn_angle:.1f} deg")
        self.post_data(turn_angle)

class DriveToBarrel(StateNode):
    def start(self, event=None):
        super().start(event)
        barrel = getattr(self.parent, 'target_barrel', None)
        if barrel is None:
            self.post_failure()
            return
        dx = barrel.pose.x - robot.pose.x
        dy = barrel.pose.y - robot.pose.y
        dist = np.sqrt(dx**2 + dy**2)
        drive = dist + 80
        #print(f"driving {drive:.1f} to barrel")
        self.post_data(drive)

class GrabBarrel(StateNode):
    def start(self, event=None):
        super().start(event)
        barrel = getattr(self.parent, 'target_barrel', None)
        if barrel is None:
            #print("grabbarrel - no target barrel")
            self.post_failure()
            return
        robot.holding = barrel
        barrel.held_by = robot
        #print(f"grabbarrel - holding {barrel}")
        self.post_completion()

class ComputeDropTurn(StateNode):  # using steps to approach location to not overshoot or run off into the distance like a problem I was getting origonally.
    def start(self, event=None):
        super().start(event)
        dx = zonex - robot.pose.x
        dy = zoney - robot.pose.y
        bearing = np.degrees(np.arctan2(dy, dx))
        turn_angle = (bearing - np.degrees(robot.pose.theta) + 180) % 360 - 180
        #print(f"turning toward zone {turn_angle:.1f} deg")
        self.post_data(turn_angle)

class ApproachDropZone(StateNode):
    distt = 100
    dists = 150

    def start(self, event=None):
        super().start(event)
        dx = zonex - robot.pose.x
        dy = zoney - robot.pose.y
        dist = np.sqrt(dx**2 + dy**2)
        #print(f"drop zone dist - {dist:.1f}")
        if dist <= self.distt:
           # print("at zone")
            self.post_success()
        else:
            step = min(dist, self.dists)
            #print(f"driving {step:.1f}")
            self.post_data(step)

class TurnToCenter(StateNode): # retuns the bot to the starting position to look for more barrels, not super accurate but doesnt need to be
    def start(self, event=None):
        super().start(event)
        dx = 0 - robot.pose.x
        dy = 0 - robot.pose.y
        bearing = np.degrees(np.arctan2(dy, dx))
        turn_angle = (bearing - np.degrees(robot.pose.theta) + 180) % 360 - 180
        #print(f"turning toward center {turn_angle:.1f} deg")
        self.post_data(turn_angle)

class ApproachOrigin(StateNode): # using steps to approach location to not overshoot or run off into the distance like a problem I was getting origonally.
    distt = 80
    dists = 150

    def start(self, event=None):
        super().start(event)
        dx = 0 - robot.pose.x
        dy = 0 - robot.pose.y
        dist = np.sqrt(dx**2 + dy**2)
        #print(f"start dist: {dist:.1f}")
        if dist <= self.distt:
            #print("Bback at start")
            self.post_success()

        else:
            step = min(dist, self.dists)
            #print(f"driving {step:.1f} toward start")
            self.post_data(step)

class FaceOriginHeading(StateNode):
    def start(self, event=None):
        super().start(event)
        current = np.degrees(robot.pose.theta)
        turn_angle = (0 - current + 180) % 360 - 180
        #print(f"current angle: {current:.1f} deg — turning {turn_angle:.1f} deg to face 0 deg")
        
        if abs(turn_angle) < 1.0:
            #print("Already at 0deg")
            self.post_completion()
        else:
            self.post_data(turn_angle)

class BarrelCollector(StateMachineProgram):

    def __init__(self):
        landmarks = {'ArucoMarker-1': Pose(-152.4 , 304.8 , 0, -pi/2),
                     'ArucoMarker-2': Pose(152.4 , 304.8 , 0 , -pi/2),
                     'ArucoMarker-7': Pose(-304.8 , -152.4 , 0 , 0),
                     'ArucoMarker-8': Pose(-304.8 , 152.4 , 0 , 0),
                     'ArucoMarker-5': Pose(152.4 , -304.8 , 0 , pi/2),
                     'ArucoMarker-6': Pose(-152.4 , -304.8 , 0 , pi/2),
                     'ArucoMarker-3': Pose(304.8 , 152.4 , 0 , -pi),
                     'ArucoMarker-4': Pose(304.8 , -152.4 , 0 , -pi)}
        pf = ParticleFilter(robot,
                            num_particles = 4000, landmarks = landmarks,
                            sensor_model = ArucoCombinedSensorModel
                            )
        super().__init__(particle_filter = pf,
                         wall_marker_dict = None,
                         speech = False,
                         launch_particle_viewer = True)

    def start(self):
        super().start()
        a1 = ArucoMarkerObj({'name':'ArucoMarker-1.a', 'id':1, 'marker':None}, x=-152.4, y=304.8, theta=-pi/2, is_fixed=True)
        a2 = ArucoMarkerObj({'name':'ArucoMarker-2.a', 'id':2, 'marker':None}, x=152.4, y=304.8, theta=-pi/2, is_fixed=True)
        a3 = ArucoMarkerObj({'name':'ArucoMarker-7.a', 'id':7, 'marker':None}, x=-304.8, y=-152.4, theta=0, is_fixed=True)
        a4 = ArucoMarkerObj({'name':'ArucoMarker-8.a', 'id':8, 'marker':None}, x=-304.8, y=152.4, theta=0, is_fixed=True)
        a5 = ArucoMarkerObj({'name':'ArucoMarker-5.a', 'id':5, 'marker':None}, x=152.4, y=-304.8, theta=pi/2, is_fixed=True)
        a6 = ArucoMarkerObj({'name':'ArucoMarker-6.a', 'id':6, 'marker':None}, x=-152.4, y=-304.8, theta=pi/2, is_fixed=True)
        a7 = ArucoMarkerObj({'name':'ArucoMarker-3.a', 'id':3, 'marker':None}, x=304.8, y=152.4, theta=-pi, is_fixed=True)
        a8 = ArucoMarkerObj({'name':'ArucoMarker-4.a', 'id':4, 'marker':None}, x=304.8, y=-152.4, theta=-pi, is_fixed=True)
        robot.world_map.objects['ArucoMarker-1.a'] = a1
        robot.world_map.objects['ArucoMarker-2.a'] = a2
        robot.world_map.objects['ArucoMarker-7.a'] = a3
        robot.world_map.objects['ArucoMarker-8.a'] = a4
        robot.world_map.objects['ArucoMarker-5.a'] = a5
        robot.world_map.objects['ArucoMarker-6.a'] = a6
        robot.world_map.objects['ArucoMarker-3.a'] = a7
        robot.world_map.objects['ArucoMarker-4.a'] = a8

    def setup(self):
        # 
        #         start: LocateBarrel()
        #         start =F=> Print("barrel not visible") =C=> Turn(45) =T(5)=> start
        #         start =D=> align
        # 
        #         align: TurnToBarrel()
        #         align =F=> start
        #         align =D=> Turn() =C=> drive
        # 
        #         drive: DriveToBarrel()
        #         drive =F=> start
        #         drive =D=> Forward() =C=> grab
        # 
        #         grab: GrabBarrel()
        #         grab =F=> start
        #         grab =C=> turnDrop
        # 
        #         turnDrop: ComputeDropTurn()
        #         turnDrop =D=> Turn() =C=> approach
        # 
        #         approach: ApproachDropZone()
        #         approach =D=> Forward() =C=> approach
        #         approach =S=> Drop() =C=> delivered
        # 
        #         delivered: Print("barrel in zone")
        #         delivered =C=> turnBack
        # 
        #         turnBack: TurnToCenter()
        #         turnBack =D=> Turn() =C=> returnHome
        # 
        #         returnHome: ApproachOrigin()
        #         returnHome =D=> Forward() =C=> returnHome
        #         returnHome =S=> faceOrigin
        # 
        #         faceOrigin: FaceOriginHeading()
        #         faceOrigin =D=> Turn() =C=> start
        #         faceOrigin =C=> start
        
        # Code generated by genfsm on Mon Jun  8 13:34:41 2026:
        
        start = LocateBarrel() .set_name("start") .set_parent(self)
        print1 = Print("barrel not visible") .set_name("print1") .set_parent(self)
        turn1 = Turn(45) .set_name("turn1") .set_parent(self)
        align = TurnToBarrel() .set_name("align") .set_parent(self)
        turn2 = Turn() .set_name("turn2") .set_parent(self)
        drive = DriveToBarrel() .set_name("drive") .set_parent(self)
        forward1 = Forward() .set_name("forward1") .set_parent(self)
        grab = GrabBarrel() .set_name("grab") .set_parent(self)
        turnDrop = ComputeDropTurn() .set_name("turnDrop") .set_parent(self)
        turn3 = Turn() .set_name("turn3") .set_parent(self)
        approach = ApproachDropZone() .set_name("approach") .set_parent(self)
        forward2 = Forward() .set_name("forward2") .set_parent(self)
        drop1 = Drop() .set_name("drop1") .set_parent(self)
        delivered = Print("barrel in zone") .set_name("delivered") .set_parent(self)
        turnBack = TurnToCenter() .set_name("turnBack") .set_parent(self)
        turn4 = Turn() .set_name("turn4") .set_parent(self)
        returnHome = ApproachOrigin() .set_name("returnHome") .set_parent(self)
        forward3 = Forward() .set_name("forward3") .set_parent(self)
        faceOrigin = FaceOriginHeading() .set_name("faceOrigin") .set_parent(self)
        turn5 = Turn() .set_name("turn5") .set_parent(self)
        
        failuretrans1 = FailureTrans() .set_name("failuretrans1")
        failuretrans1 .add_sources(start) .add_destinations(print1)
        
        completiontrans1 = CompletionTrans() .set_name("completiontrans1")
        completiontrans1 .add_sources(print1) .add_destinations(turn1)
        
        timertrans1 = TimerTrans(5) .set_name("timertrans1")
        timertrans1 .add_sources(turn1) .add_destinations(start)
        
        datatrans1 = DataTrans() .set_name("datatrans1")
        datatrans1 .add_sources(start) .add_destinations(align)
        
        failuretrans2 = FailureTrans() .set_name("failuretrans2")
        failuretrans2 .add_sources(align) .add_destinations(start)
        
        datatrans2 = DataTrans() .set_name("datatrans2")
        datatrans2 .add_sources(align) .add_destinations(turn2)
        
        completiontrans2 = CompletionTrans() .set_name("completiontrans2")
        completiontrans2 .add_sources(turn2) .add_destinations(drive)
        
        failuretrans3 = FailureTrans() .set_name("failuretrans3")
        failuretrans3 .add_sources(drive) .add_destinations(start)
        
        datatrans3 = DataTrans() .set_name("datatrans3")
        datatrans3 .add_sources(drive) .add_destinations(forward1)
        
        completiontrans3 = CompletionTrans() .set_name("completiontrans3")
        completiontrans3 .add_sources(forward1) .add_destinations(grab)
        
        failuretrans4 = FailureTrans() .set_name("failuretrans4")
        failuretrans4 .add_sources(grab) .add_destinations(start)
        
        completiontrans4 = CompletionTrans() .set_name("completiontrans4")
        completiontrans4 .add_sources(grab) .add_destinations(turnDrop)
        
        datatrans4 = DataTrans() .set_name("datatrans4")
        datatrans4 .add_sources(turnDrop) .add_destinations(turn3)
        
        completiontrans5 = CompletionTrans() .set_name("completiontrans5")
        completiontrans5 .add_sources(turn3) .add_destinations(approach)
        
        datatrans5 = DataTrans() .set_name("datatrans5")
        datatrans5 .add_sources(approach) .add_destinations(forward2)
        
        completiontrans6 = CompletionTrans() .set_name("completiontrans6")
        completiontrans6 .add_sources(forward2) .add_destinations(approach)
        
        successtrans1 = SuccessTrans() .set_name("successtrans1")
        successtrans1 .add_sources(approach) .add_destinations(drop1)
        
        completiontrans7 = CompletionTrans() .set_name("completiontrans7")
        completiontrans7 .add_sources(drop1) .add_destinations(delivered)
        
        completiontrans8 = CompletionTrans() .set_name("completiontrans8")
        completiontrans8 .add_sources(delivered) .add_destinations(turnBack)
        
        datatrans6 = DataTrans() .set_name("datatrans6")
        datatrans6 .add_sources(turnBack) .add_destinations(turn4)
        
        completiontrans9 = CompletionTrans() .set_name("completiontrans9")
        completiontrans9 .add_sources(turn4) .add_destinations(returnHome)
        
        datatrans7 = DataTrans() .set_name("datatrans7")
        datatrans7 .add_sources(returnHome) .add_destinations(forward3)
        
        completiontrans10 = CompletionTrans() .set_name("completiontrans10")
        completiontrans10 .add_sources(forward3) .add_destinations(returnHome)
        
        successtrans2 = SuccessTrans() .set_name("successtrans2")
        successtrans2 .add_sources(returnHome) .add_destinations(faceOrigin)
        
        datatrans8 = DataTrans() .set_name("datatrans8")
        datatrans8 .add_sources(faceOrigin) .add_destinations(turn5)
        
        completiontrans11 = CompletionTrans() .set_name("completiontrans11")
        completiontrans11 .add_sources(turn5) .add_destinations(start)
        
        completiontrans12 = CompletionTrans() .set_name("completiontrans12")
        completiontrans12 .add_sources(faceOrigin) .add_destinations(start)
        
        return self
