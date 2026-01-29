import React from 'react';
import { Snackbar, Alert, AlertTitle, Slide, SlideProps } from '@mui/material';
import { useAppDispatch, useAppSelector } from '../../store';
import { removeNotification } from '../../store/slices/uiSlice';

function SlideTransition(props: SlideProps) {
  return <Slide {...props} direction="left" />;
}

interface NotificationProviderProps {
  children: React.ReactNode;
}

const NotificationProvider: React.FC<NotificationProviderProps> = ({ children }) => {
  const dispatch = useAppDispatch();
  const notifications = useAppSelector((state) => state.ui.notifications);
  
  // Show only the most recent notification that should auto-hide
  const currentNotification = notifications.find(n => n.autoHide !== false);

  const handleClose = (notificationId?: string) => {
    if (notificationId) {
      dispatch(removeNotification(notificationId));
    }
  };

  return (
    <>
      {children}
      
      {currentNotification && (
        <Snackbar
          open={true}
          autoHideDuration={currentNotification.autoHide === false ? null : 6000}
          onClose={() => handleClose(currentNotification.id)}
          anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
          TransitionComponent={SlideTransition}
          sx={{ mt: 8 }} // Account for AppBar height
        >
          <Alert
            onClose={() => handleClose(currentNotification.id)}
            severity={currentNotification.type}
            variant="filled"
            sx={{ width: '100%', minWidth: 300 }}
          >
            {currentNotification.title && (
              <AlertTitle>{currentNotification.title}</AlertTitle>
            )}
            {currentNotification.message}
          </Alert>
        </Snackbar>
      )}
    </>
  );
};

export default NotificationProvider;