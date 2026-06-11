from aim_fsm import *
import numpy as np

zonex = 225 # back right corner from starting pose set as goal location
zoney = 225
color = OrangeBarrelObj # team color

######################################
#---Movement and Barrel Collection---#
######################################

class LocateUnsecureBarrel(StateNode):
    def __init__(self, tracker_program):
        # Pass the main program reference so we can access its persistent registry
        self.tracker = tracker_program
        super().__init__()

    def start(self, event=None):
        super().start(event)
        closest = None
        best_dist = float('inf')
        for obj in self.tracker.uncap:
            if isinstance(obj, color) and not obj.is_missing:
                dx = obj.pose.x - robot.pose.x 
                dy = obj.pose.y - robot.pose.y
                dist = np.sqrt(dx**2 + dy**2)# finding distance same as rest
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
        if self.robot.holding is not None:
            if isinstance(self.robot.holding , OrangeBarrelObj):
                prevHold = self.robot.holding
                self.robot.holding = barrel
                #Remove Previous Barrel
            else:
                self.post_failure()
        else:
            self.post_failure()
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
        if (robot.pose.x > 350 or robot.pose.x < -350 or robot.pose.y > 350 or robot.pose.y < -350):
            self.robot.particle_filter.delocalize()
            self.post_failure()
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

#######################
#---Barrel Tracking---#
#######################

class barrelGamePosition(StateNode):
    def __init__(self, tracker_program):
        # Pass the main program reference so we can access its persistent registry
        self.tracker = tracker_program
        super().__init__()
    
    def start(self, event=None):
        blue_win_thresh = -150
        orange_win_thresh = 150
        super().start(event)

        ### check for all barrels on the field ###
        visible_orange = []
        visible_blue = []
        
        for obj in robot.world_map.objects.values():
            if isinstance(obj, BarrelObj) and getattr(obj, 'is_visible', False):
                if isinstance(obj, OrangeBarrelObj):
                    visible_orange.append(obj)
                elif isinstance(obj, BlueBarrelObj):
                    visible_blue.append(obj)

        ### orange tracking operations
        for obj in visible_orange:
            # if this obj is already tracked, skip it
            if obj in self.tracker.orange_registry.values():
                continue
                
            # if it isn't being tracked, look for an empty spot in the list or an unseen barrel to assign it to
            reclaimed = False
            for slot, registered_obj in self.tracker.orange_registry.items():
                if registered_obj is None or getattr(registered_obj, 'is_missing', True):
                    self.tracker.orange_registry[slot] = obj
                    reclaimed = True
                    break
            
            # if everything is full just ignore it, there shouldn't be more than 3 barrels on the field
            if not reclaimed and len(self.tracker.orange_registry) < 3:
                slot_name = f"orange_{len(self.tracker.orange_registry) + 1}"
                self.tracker.orange_registry[slot_name] = obj

        ### same thing for the blue barrels
        for obj in visible_blue:
            if obj in self.tracker.blue_registry.values():
                continue
                
            reclaimed = False
            for slot, registered_obj in self.tracker.blue_registry.items():
                if registered_obj is None or getattr(registered_obj, 'is_missing', True):
                    self.tracker.blue_registry[slot] = obj
                    reclaimed = True
                    break
            
            if not reclaimed and len(self.tracker.blue_registry) < 3:
                slot_name = f"blue_{len(self.tracker.blue_registry) + 1}"
                self.tracker.blue_registry[slot_name] = obj

        ### sort into the 3 different lists
        self.tracker.orange_cap.clear()
        self.tracker.blue_cap.clear()
        self.tracker.uncap.clear()

        ### check if orange barrels are beyond their captured threshold 
        for obj in self.tracker.orange_registry.values():
            if obj is not None:
                if obj.pose.x > orange_win_thresh and obj.pose.y > orange_win_thresh:
                    self.tracker.orange_cap.append(obj)
                else:
                    self.tracker.uncap.append(obj)
                    

        ### check if the blue barrels are beyond their captured threshold
        for obj in self.tracker.blue_registry.values():
            if obj is not None:
                if obj.pose.x < blue_win_thresh and obj.pose.y < blue_win_thresh:
                    self.tracker.blue_cap.append(obj)
                else:
                    self.tracker.uncap.append(obj)

        ### posting the data
        # calculated the number of barrels being tracked
        total_tracked = len([x for x in self.tracker.orange_registry.values() if x is not None]) + \
                        len([x for x in self.tracker.blue_registry.values() if x is not None])

        if total_tracked == 6:
            self.post_data((self.tracker.orange_cap, self.tracker.blue_cap, self.tracker.uncap))
        else:
            print(f"Still initializing... Tracked unique barrels: {total_tracked}/6")
            self.post_failure()


### helper class to print the results for the video demo
class PrintResultsNode(StateNode):
    def start(self, event=None):
        super().start(event)
        if event and event.data:
            orange, blue, uncaptured = event.data
            print("\n--- Barrel Game Update ---")
            print(f"Captured Orange: {len(orange)} barrels")
            print(f"Captured Blue:   {len(blue)} barrels")
            print(f"Uncaptured:      {len(uncaptured)} barrels")
            print("--------------------------\n")
        self.post_success()

class ClearWorld(StateNode):
    def start(self , event=None):
        super().start(event)
        robot.world_map.clear()
        self.post_success

class PrintPosition(StateNode):
    def start(self , event=None):
        super().start(event)
        print("Robot X: " ,  robot.pose.x , "| Robot y: " , robot.pose.y)


##################
#---Main Class---#
##################

class PacifistBot(StateMachineProgram):
    def __init__(self):

        # tracking lists
        self.orange_cap = []
        self.blue_cap = []
        self.uncap = []
        
        # registries to act as permanent trackers for barrels to prevent phantom barrels
        self.orange_registry = {"orange_1": None, "orange_2": None, "orange_3": None}
        self.blue_registry = {"blue_1": None, "blue_2": None, "blue_3": None}
        
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
        #         Localize: Forward(50) =C=> Turn(135) =C=> ClearWorld() =T(1)=> InitBarrels
        # 
        #         InitBarrels: barrelGamePosition(self)
        #         InitBarrels =D=> PrintResultsNode() =S=> {UpdateBarrels , locate}
        #         InitBarrels =F=> Turn(-45) =T(4)=> InitBarrels
        # 
        #         UpdateBarrels: barrelGamePosition(self)
        #         UpdateBarrels =D=> PrintResultsNode() =T(1)=> UpdateBarrels
        #         UpdateBarrels =F=> Print() =T(1)=> UpdateBarrels
        # 
        #         locate: LocateUnsecureBarrel(self)
        #         locate =F=> Print("Couldn't Locate Barrel") =C=> Remap
        #         locate =D=> align
        # 
        #         Remap: Turn(45) =T(3)=> Turn(45) =T(3)=> Turn(45) =T(3)=> Turn(45) =T(3)=> Turn(45) =T(3)=> Turn(45) =T(3)=> Turn(45) =T(3)=> Turn(45) =T(3)=> locate
        # 
        #         align: TurnToBarrel()
        #         align =F=> locate
        #         align =D=> Turn() =T(5)=> drive
        # 
        #         drive: DriveToBarrel()
        #         drive =F=> locate
        #         drive =D=> Forward() =T(5)=> grab
        # 
        #         grab: GrabBarrel()
        #         grab =F=> Kick() =C=> Remap
        #         grab =C=> turnDrop
        # 
        #         turnDrop: ComputeDropTurn()
        #         turnDrop =D=> Turn() =T(5)=> approach
        # 
        #         approach: ApproachDropZone()
        #         approach =D=> Forward() =C=> approach
        #         approach =F=> Remap
        #         approach =S=> Kick() =C=> Print("Barrel Dropped In Zone") =C=> Forward(-50) =T(2)=> DropZoneLocate
        # 
        #         DropZoneLocate: Turn(45) =T(3)=> Turn(45) =T(3)=> Turn(45) =T(3)=> Turn(45) =T(3)=> Turn(45) =T(3)=> Turn(45) =T(3)=> locate
        
        # Code generated by genfsm on Tue Jun  9 10:18:37 2026:
        
        Localize = Forward(50) .set_name("Localize") .set_parent(self)
        turn1 = Turn(135) .set_name("turn1") .set_parent(self)
        clearworld1 = ClearWorld() .set_name("clearworld1") .set_parent(self)
        InitBarrels = barrelGamePosition(self) .set_name("InitBarrels") .set_parent(self)
        printresultsnode1 = PrintResultsNode() .set_name("printresultsnode1") .set_parent(self)
        turn2 = Turn(-45) .set_name("turn2") .set_parent(self)
        UpdateBarrels = barrelGamePosition(self) .set_name("UpdateBarrels") .set_parent(self)
        printresultsnode2 = PrintResultsNode() .set_name("printresultsnode2") .set_parent(self)
        print1 = Print() .set_name("print1") .set_parent(self)
        locate = LocateUnsecureBarrel(self) .set_name("locate") .set_parent(self)
        print2 = Print("Couldn't Locate Barrel") .set_name("print2") .set_parent(self)
        Remap = Turn(45) .set_name("Remap") .set_parent(self)
        turn3 = Turn(45) .set_name("turn3") .set_parent(self)
        turn4 = Turn(45) .set_name("turn4") .set_parent(self)
        turn5 = Turn(45) .set_name("turn5") .set_parent(self)
        turn6 = Turn(45) .set_name("turn6") .set_parent(self)
        turn7 = Turn(45) .set_name("turn7") .set_parent(self)
        turn8 = Turn(45) .set_name("turn8") .set_parent(self)
        turn9 = Turn(45) .set_name("turn9") .set_parent(self)
        align = TurnToBarrel() .set_name("align") .set_parent(self)
        turn10 = Turn() .set_name("turn10") .set_parent(self)
        drive = DriveToBarrel() .set_name("drive") .set_parent(self)
        forward1 = Forward() .set_name("forward1") .set_parent(self)
        grab = GrabBarrel() .set_name("grab") .set_parent(self)
        kick1 = Kick() .set_name("kick1") .set_parent(self)
        turnDrop = ComputeDropTurn() .set_name("turnDrop") .set_parent(self)
        turn11 = Turn() .set_name("turn11") .set_parent(self)
        approach = ApproachDropZone() .set_name("approach") .set_parent(self)
        forward2 = Forward() .set_name("forward2") .set_parent(self)
        kick2 = Kick() .set_name("kick2") .set_parent(self)
        print3 = Print("Barrel Dropped In Zone") .set_name("print3") .set_parent(self)
        forward3 = Forward(-50) .set_name("forward3") .set_parent(self)
        DropZoneLocate = Turn(45) .set_name("DropZoneLocate") .set_parent(self)
        turn12 = Turn(45) .set_name("turn12") .set_parent(self)
        turn13 = Turn(45) .set_name("turn13") .set_parent(self)
        turn14 = Turn(45) .set_name("turn14") .set_parent(self)
        turn15 = Turn(45) .set_name("turn15") .set_parent(self)
        turn16 = Turn(45) .set_name("turn16") .set_parent(self)
        
        completiontrans1 = CompletionTrans() .set_name("completiontrans1")
        completiontrans1 .add_sources(Localize) .add_destinations(turn1)
        
        completiontrans2 = CompletionTrans() .set_name("completiontrans2")
        completiontrans2 .add_sources(turn1) .add_destinations(clearworld1)
        
        timertrans1 = TimerTrans(1) .set_name("timertrans1")
        timertrans1 .add_sources(clearworld1) .add_destinations(InitBarrels)
        
        datatrans1 = DataTrans() .set_name("datatrans1")
        datatrans1 .add_sources(InitBarrels) .add_destinations(printresultsnode1)
        
        successtrans1 = SuccessTrans() .set_name("successtrans1")
        successtrans1 .add_sources(printresultsnode1) .add_destinations(UpdateBarrels,locate)
        
        failuretrans1 = FailureTrans() .set_name("failuretrans1")
        failuretrans1 .add_sources(InitBarrels) .add_destinations(turn2)
        
        timertrans2 = TimerTrans(4) .set_name("timertrans2")
        timertrans2 .add_sources(turn2) .add_destinations(InitBarrels)
        
        datatrans2 = DataTrans() .set_name("datatrans2")
        datatrans2 .add_sources(UpdateBarrels) .add_destinations(printresultsnode2)
        
        timertrans3 = TimerTrans(1) .set_name("timertrans3")
        timertrans3 .add_sources(printresultsnode2) .add_destinations(UpdateBarrels)
        
        failuretrans2 = FailureTrans() .set_name("failuretrans2")
        failuretrans2 .add_sources(UpdateBarrels) .add_destinations(print1)
        
        timertrans4 = TimerTrans(1) .set_name("timertrans4")
        timertrans4 .add_sources(print1) .add_destinations(UpdateBarrels)
        
        failuretrans3 = FailureTrans() .set_name("failuretrans3")
        failuretrans3 .add_sources(locate) .add_destinations(print2)
        
        completiontrans3 = CompletionTrans() .set_name("completiontrans3")
        completiontrans3 .add_sources(print2) .add_destinations(Remap)
        
        datatrans3 = DataTrans() .set_name("datatrans3")
        datatrans3 .add_sources(locate) .add_destinations(align)
        
        timertrans5 = TimerTrans(3) .set_name("timertrans5")
        timertrans5 .add_sources(Remap) .add_destinations(turn3)
        
        timertrans6 = TimerTrans(3) .set_name("timertrans6")
        timertrans6 .add_sources(turn3) .add_destinations(turn4)
        
        timertrans7 = TimerTrans(3) .set_name("timertrans7")
        timertrans7 .add_sources(turn4) .add_destinations(turn5)
        
        timertrans8 = TimerTrans(3) .set_name("timertrans8")
        timertrans8 .add_sources(turn5) .add_destinations(turn6)
        
        timertrans9 = TimerTrans(3) .set_name("timertrans9")
        timertrans9 .add_sources(turn6) .add_destinations(turn7)
        
        timertrans10 = TimerTrans(3) .set_name("timertrans10")
        timertrans10 .add_sources(turn7) .add_destinations(turn8)
        
        timertrans11 = TimerTrans(3) .set_name("timertrans11")
        timertrans11 .add_sources(turn8) .add_destinations(turn9)
        
        timertrans12 = TimerTrans(3) .set_name("timertrans12")
        timertrans12 .add_sources(turn9) .add_destinations(locate)
        
        failuretrans4 = FailureTrans() .set_name("failuretrans4")
        failuretrans4 .add_sources(align) .add_destinations(locate)
        
        datatrans4 = DataTrans() .set_name("datatrans4")
        datatrans4 .add_sources(align) .add_destinations(turn10)
        
        timertrans13 = TimerTrans(5) .set_name("timertrans13")
        timertrans13 .add_sources(turn10) .add_destinations(drive)
        
        failuretrans5 = FailureTrans() .set_name("failuretrans5")
        failuretrans5 .add_sources(drive) .add_destinations(locate)
        
        datatrans5 = DataTrans() .set_name("datatrans5")
        datatrans5 .add_sources(drive) .add_destinations(forward1)
        
        timertrans14 = TimerTrans(5) .set_name("timertrans14")
        timertrans14 .add_sources(forward1) .add_destinations(grab)
        
        failuretrans6 = FailureTrans() .set_name("failuretrans6")
        failuretrans6 .add_sources(grab) .add_destinations(kick1)
        
        completiontrans4 = CompletionTrans() .set_name("completiontrans4")
        completiontrans4 .add_sources(kick1) .add_destinations(Remap)
        
        completiontrans5 = CompletionTrans() .set_name("completiontrans5")
        completiontrans5 .add_sources(grab) .add_destinations(turnDrop)
        
        datatrans6 = DataTrans() .set_name("datatrans6")
        datatrans6 .add_sources(turnDrop) .add_destinations(turn11)
        
        timertrans15 = TimerTrans(5) .set_name("timertrans15")
        timertrans15 .add_sources(turn11) .add_destinations(approach)
        
        datatrans7 = DataTrans() .set_name("datatrans7")
        datatrans7 .add_sources(approach) .add_destinations(forward2)
        
        completiontrans6 = CompletionTrans() .set_name("completiontrans6")
        completiontrans6 .add_sources(forward2) .add_destinations(approach)
        
        failuretrans7 = FailureTrans() .set_name("failuretrans7")
        failuretrans7 .add_sources(approach) .add_destinations(Remap)
        
        successtrans2 = SuccessTrans() .set_name("successtrans2")
        successtrans2 .add_sources(approach) .add_destinations(kick2)
        
        completiontrans7 = CompletionTrans() .set_name("completiontrans7")
        completiontrans7 .add_sources(kick2) .add_destinations(print3)
        
        completiontrans8 = CompletionTrans() .set_name("completiontrans8")
        completiontrans8 .add_sources(print3) .add_destinations(forward3)
        
        timertrans16 = TimerTrans(2) .set_name("timertrans16")
        timertrans16 .add_sources(forward3) .add_destinations(DropZoneLocate)
        
        timertrans17 = TimerTrans(3) .set_name("timertrans17")
        timertrans17 .add_sources(DropZoneLocate) .add_destinations(turn12)
        
        timertrans18 = TimerTrans(3) .set_name("timertrans18")
        timertrans18 .add_sources(turn12) .add_destinations(turn13)
        
        timertrans19 = TimerTrans(3) .set_name("timertrans19")
        timertrans19 .add_sources(turn13) .add_destinations(turn14)
        
        timertrans20 = TimerTrans(3) .set_name("timertrans20")
        timertrans20 .add_sources(turn14) .add_destinations(turn15)
        
        timertrans21 = TimerTrans(3) .set_name("timertrans21")
        timertrans21 .add_sources(turn15) .add_destinations(turn16)
        
        timertrans22 = TimerTrans(3) .set_name("timertrans22")
        timertrans22 .add_sources(turn16) .add_destinations(locate)
        
        return self
