import type { NotificationChannel } from "@/types";
import { ToastChannel } from "./ToastChannel";
import { BannerChannel } from "./BannerChannel";
import { DrawerChannel } from "./DrawerChannel";
import { ModalChannel } from "./ModalChannel";

/**
 * 채널 → 컴포넌트 레지스트리 (Factory 의 렌더링 측면).
 * severity 가 NotificationFactory.channelOf 로 채널을 정하면, 그 채널의 컴포넌트가
 * 여기 등록된 매핑을 통해 렌더링된다. 새 채널 추가 = 이 맵에 한 줄 등록(OCP).
 */
const CHANNEL_COMPONENTS: Record<NotificationChannel, React.FC> = {
  toast: ToastChannel,
  banner: BannerChannel,
  drawer: DrawerChannel,
  modal: ModalChannel,
};

/** 모든 알림 채널을 한 번에 마운트한다. AppLayout 에서 1회 렌더. */
export function NotificationOutlet() {
  return (
    <>
      {Object.entries(CHANNEL_COMPONENTS).map(([channel, Component]) => (
        <Component key={channel} />
      ))}
    </>
  );
}
