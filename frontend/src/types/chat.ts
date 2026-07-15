/** 채팅 도메인 타입. */

export interface ChatUser {
  readonly id: string;
  readonly name: string;
  readonly role: string;
  readonly department: string;
}

export interface ChatRoom {
  readonly id: string;
  readonly memberIds: string;
  readonly createdAt: string;
}

export interface ChatMessage {
  readonly id: string;
  readonly roomId: string;
  readonly senderId: string;
  readonly senderName: string;
  readonly body: string;
  readonly createdAt: string;
}
