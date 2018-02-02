import rospy
from std_msgs.msg import String


def periodic_publish():
	pub = rospy.Publisher('Sree_Hello_World_topic', String, queue_size=10)
	rospy.init_node('SreePublisher_node', anonymous=True)
	r = rospy.Rate(10)
	while not rospy.is_shutdown():
		pub.publish("hello World")
		r.sleep()

if __name__ == "__main__":
    periodic_publish()

