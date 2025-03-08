import React from 'react';
import { Box, Typography } from '@mui/material';
import styled from 'styled-components';

const NeonFooter = styled(Box)`
  background: rgba(13, 13, 13, 0.8); /* Semi-transparent dark overlay */
  backdrop-filter: blur(10px);
  color: white;
  text-align: center;
  padding: '10px';
  margin-top: 'auto';
`;

function Footer() {
    return (
        <NeonFooter component="footer">
            <Typography variant="body2">© 2025 SheGuard. All rights reserved.</Typography>
        </NeonFooter>
    );
}

export default Footer;