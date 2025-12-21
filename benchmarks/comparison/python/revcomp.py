# The Computer Language Benchmarks Game
# https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
#
# Contributed by Jeremy Zerfas.

from multiprocessing import Queue, Semaphore, Condition, Value, Process
from sys import stdin, stdout
from os import cpu_count, read, write

def process_Sequences(info_For_Remaining_Sequences_Data_To_Process
  , stdin_File_Descriptor, CPU_Cores_Available_For_Processing_Sequences
  , sequence_Was_Written_Condition, next_Sequence_Number_To_Output):

	COMPLEMENT_LOOKUP=(
	  b"          \n                                                     "
	  b" TVGH  CD  M KN   YSAABW R       TVGH  CD  M KN   YSAABW R      "
	  b"                                                                "
	  b"                                                                ")

	READ_SIZE=65536
	LINE_LENGTH=60

	while True:
		remaining_Sequences_Data_To_Process, sequence_Number= \
		  info_For_Remaining_Sequences_Data_To_Process.get()

		if remaining_Sequences_Data_To_Process is None:
			info_For_Remaining_Sequences_Data_To_Process.put((None, None))
			CPU_Cores_Available_For_Processing_Sequences.release()
			break

		sequence=bytearray()
		sequence+=remaining_Sequences_Data_To_Process[0:1]
		remaining_Sequences_Data_To_Process= \
		  remaining_Sequences_Data_To_Process[1:]

		while True:
			if b">" in remaining_Sequences_Data_To_Process:
				preceding_Bytes, _, following_Bytes= \
				  remaining_Sequences_Data_To_Process.partition(b">")
				sequence+=preceding_Bytes
				remaining_Sequences_Data_To_Process=b">"+following_Bytes
				break

			sequence+=remaining_Sequences_Data_To_Process
			remaining_Sequences_Data_To_Process=read(stdin_File_Descriptor, READ_SIZE)

			if not remaining_Sequences_Data_To_Process:
				break

		if remaining_Sequences_Data_To_Process:
			info_For_Remaining_Sequences_Data_To_Process.put( \
			  (remaining_Sequences_Data_To_Process, sequence_Number+1))

			if sequence_Number>0 \
			  and CPU_Cores_Available_For_Processing_Sequences.acquire(False):
				Process(target=process_Sequences, args=(
				  info_For_Remaining_Sequences_Data_To_Process
				  , stdin_File_Descriptor
				  , CPU_Cores_Available_For_Processing_Sequences
				  , sequence_Was_Written_Condition
				  , next_Sequence_Number_To_Output)).start()
		else:
			info_For_Remaining_Sequences_Data_To_Process.put((None, None))

		if sequence_Number>0:
			sequence_Header, _, temporary_Sequence_Data= \
			  sequence.partition(b"\n")
			del sequence

			if len(temporary_Sequence_Data)%(LINE_LENGTH+1)==0:
				modified_Sequence_Data= \
				  temporary_Sequence_Data.translate(COMPLEMENT_LOOKUP)
				modified_Sequence_Data.reverse()
				modified_Sequence_Data+=b"\n"
			else:
				temporary_Sequence_Data= \
				  temporary_Sequence_Data.translate(COMPLEMENT_LOOKUP, b"\n")
				temporary_Sequence_Data.reverse()
				modified_Sequence_Data=bytearray(b"\n")
				for i in range(0, len(temporary_Sequence_Data), LINE_LENGTH):
					modified_Sequence_Data+= \
					  temporary_Sequence_Data[i:i+LINE_LENGTH]
					modified_Sequence_Data+=b"\n"

			del temporary_Sequence_Data

			with sequence_Was_Written_Condition:
				while next_Sequence_Number_To_Output.value<sequence_Number:
					sequence_Was_Written_Condition.wait()

				write(stdout.fileno(), sequence_Header)
				write(stdout.fileno(), modified_Sequence_Data)

				next_Sequence_Number_To_Output.value+=1
				sequence_Was_Written_Condition.notify_all()

			del modified_Sequence_Data

if __name__=="__main__":
	info_For_Remaining_Sequences_Data_To_Process=Queue()
	info_For_Remaining_Sequences_Data_To_Process.put((b"", 0))

	stdin_File_Descriptor=stdin.fileno()

	CPU_Cores_Available_For_Processing_Sequences=Semaphore((cpu_count() or 1)-1)

	sequence_Was_Written_Condition=Condition()
	next_Sequence_Number_To_Output=Value("L", 1)

	process_Sequences(info_For_Remaining_Sequences_Data_To_Process
	  , stdin_File_Descriptor, CPU_Cores_Available_For_Processing_Sequences
	  , sequence_Was_Written_Condition, next_Sequence_Number_To_Output)

	for i in range(cpu_count() or 1):
		CPU_Cores_Available_For_Processing_Sequences.acquire()
